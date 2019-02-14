import os,sys
sys.path.append(os.path.dirname(__file__))
sys.path.append(os.path.join(os.path.dirname(__file__),'..'))
import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow.contrib.rnn as rnn
import tensorflow.contrib.layers as layers

from base_trainer import *
from dataprepare import *



class EncoderDecoderModel(object):
    def __init__(self, dp, _path_prefix):
        self.dp = dp
        
        self.seed = 9
        self.encoder_layer_num = 1
        self.x_feature_num = self.dp.get_x_feature_num()

        self.encoder_num_units = 64

        self.y_feature_num = self.dp.get_y_feature_num()
        # self.decoder_label_step = 5

        self.lr = 0.01
        self.l1_loss_ratio = 0.002
        # self.l1_loss_ratio = 0
        self.l2_loss_ratio = 0
        self.clip_grads_max = 100
        self.input_keep_prob = 1.0
        self.output_keep_prob = 1.0
        self.path_prefix = _path_prefix

    def default_init(self):
        # replica of tf.glorot_uniform_initializer(seed=seed)
        return tf.variance_scaling_initializer(scale=1.0,
                                               mode="fan_in",
                                               distribution='normal',
                                               seed=self.seed)

    def make_encoder(self):
        # x_input [batch, step, x_Feature_num]
        self.batch_size = tf.placeholder(tf.int32, [], name='batch_size')
        self.x_input = tf.placeholder(tf.float64, shape=(None, None, self.x_feature_num))

        def get_rnn_cell():
            rnn_cell = tf.contrib.rnn.BasicLSTMCell(num_units=self.encoder_num_units, 
                                                    state_is_tuple=True)
            rnn_cell = tf.contrib.rnn.DropoutWrapper(cell=rnn_cell, input_keep_prob=self.input_keep_prob, 
                                    output_keep_prob=self.output_keep_prob)
            return rnn_cell

        with tf.name_scope('multi_layer_lstm') as vs_multi_lstm:
            mrnn_cell = tf.contrib.rnn.MultiRNNCell([get_rnn_cell() for i in range(self.encoder_layer_num)], state_is_tuple=True)
            self._inital_state = mrnn_cell.zero_state(self.batch_size, tf.float64)
        # self.encoder_c_size = rnn_cell.state_size.c
        # self.encoder_h_size = rnn_cell.state_size.h
        # # encoder c/h [batch, c/h size]
        # self.encoder_c_in = tf.placeholder(tf.float64, shape=(None, self.encoder_c_size))
        # self.encoder_h_in = tf.placeholder(tf.float64, shape=(None, self.encoder_h_size))
        with tf.variable_scope('encoder') as vs_encoder:
            encoder_output, encoder_state = tf.nn.dynamic_rnn(
                mrnn_cell,
                self.x_input,
                initial_state=self._inital_state,
                time_major=False
            )

        # output [batch,step,encoder_unit_num]
        # state: [(c,h)] * layer_num
        return encoder_output, encoder_state[-1]
        

    def make_decoder(self,encoder_state, test_window):
        """
        Parameters:
        ----------
        pre_label: shape [batch,1]
        """
        with tf.variable_scope('scope_decoder') as vs_decoder:
            # y_input: [batch, step, y_feature_num]
            self.y_input = tf.placeholder(tf.float64, shape=(None, None, self.y_feature_num))
            rnn_cell = tf.contrib.rnn.BasicLSTMCell(num_units=self.encoder_num_units, 
                                                        state_is_tuple=True)
            y_input_bytime = tf.transpose(self.y_input, [1,0,2])
            
            # deocder时刻的每个样本的实际值label shape=[batch, decoder_raw_label_feature]
            # decoder_raw_label_feature表示将前多少步的label值作为新的输入
            self.pre_label = tf.placeholder(tf.float64, shape=(None,self.dp.raw_label_input_size))

            # 利用prelabel 构造相关特征，从而用于decoder训练
            # pre_label=[batch, decoder_raw_label_feature]
            # output:[batch, decoder_input_feature_size]
            def make_pre_label_features(pre_label):
                return pre_label

            # 更新用于构造prelabel的原始特征值，因为随着时间步的推移，预测值逐渐在取代掉以前的实际值
            # prev_output: [batch,1]
            # pre_label:[batch, decoder_raw_label_feature]
            # output:[batch, decoder_raw_label_feature]
            def update_pre_label(prev_output, pre_label):
                if prev_output is None:
                    return pre_label
                return tf.concat([pre_label[:,1:], prev_output], axis=1)

            def cond_func(time, prev_output, pre_label, prev_state, array_targets):
                return time < test_window

            def project_output(tensor):
                return tf.layers.dense(tensor, 1, name='decoder_output_proj',
                                        kernel_initializer=self.default_init())
            
            def loop_func(time, prev_output, pre_label, prev_state, array_targets):
                # TODO: reuse???
                # input for current step [batch,y_input_size]
                y_features_input_i = y_input_bytime[time]
                pre_label = update_pre_label(prev_output, pre_label)
                prev_output_features = make_pre_label_features(pre_label)
                next_input = tf.concat([prev_output_features,y_features_input_i], axis=1)
                output, state = rnn_cell(next_input, prev_state)
                pj_output = project_output(output)
                array_targets = array_targets.write(time, pj_output)
                return time+1, pj_output, pre_label, state, array_targets

            loop_init = [tf.constant(0, dtype=tf.int32),
                        self.pre_label[:,-1:],
                        self.pre_label,
                        encoder_state,
                        tf.TensorArray(dtype=tf.float64, size=test_window)]
            # Run the loop
            _time, _p_output, _p_label, _p_state, targets_ta = tf.while_loop(cond_func, loop_func, loop_init)

            targets = targets_ta.stack()
            targets = tf.transpose(targets, [1,0,2])
            return targets

    def get_loss(self, targets):
        """
        Parameters:
        -----------
        targets: tensor [batch, step, 1]
        """

        # labels [batch, step, 1]
        train_vars  = tf.trainable_variables() 
        lossL2 = tf.add_n([ tf.nn.l2_loss(v) for v in train_vars ]) * self.l2_loss_ratio
        lossL1 = tf.add_n([tf.reduce_mean(tf.abs(v))for v in train_vars]) * self.l1_loss_ratio
        self.labels = tf.placeholder(tf.float64, shape=(None, None, 1))
        labels_reshape = tf.reshape(self.labels, (-1, 1))
        targets_reshape = tf.reshape(targets, (-1, 1))
        
        loss = tf.reduce_mean(tf.pow(labels_reshape - targets_reshape, 2))
        loss = loss + lossL1 + lossL2
        
        # loss = 100*tf.reduce_mean(tf.abs(labels_reshape - targets_reshape))
        return loss

    # def get_init_lstm_state(self, batch_size):
    #     return np.zeros((batch_size, self.encoder_c_size)), np.zeros((batch_size, self.encoder_h_size))

    def init_network(self):
        encoder_output, encoder_state = self.make_encoder()
        self.targets = self.make_decoder(encoder_state, self.dp.test_window)
        self.loss = self.get_loss(self.targets)

        global_step = tf.Variable(0,trainable=False)
        lr_decay = self.lr
        self.opt = tf.train.AdamOptimizer(learning_rate=lr_decay)

        self.train_variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
        grads_vars = self.opt.compute_gradients(self.loss, self.train_variables)
        # clip the gradients ensure the gradient not explode
        for i, (g,v) in enumerate(grads_vars):
            if g is not None:
                grads_vars[i] = (tf.clip_by_norm(g, self.clip_grads_max), v)
        self.train_op = self.opt.apply_gradients(grads_vars,global_step=global_step)

        self.sess = tf.Session()
        init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
        self.sess.run(init_op)
        print('init encoder-decoder network done.')

    def train(self):
        # train_lstm_c_in,train_lstm_h_in = self.get_init_lstm_state(self.dp.batch_size)
        cnt = 0
        for feature_x, feature_y, label, pre_label in self.dp.train_batch():
            feed_dict = {
                self.x_input: feature_x,
                self.y_input: feature_y,
                self.labels: np.expand_dims(label, -1),
                self.pre_label:  pre_label,
                self.batch_size: feature_x.shape[0] # train batch size
            }
            # print('shape:', feature_x.shape, feature_y.shape, label.shape, pre_label.shape)
            _,loss_value,logits_value = self.sess.run([self.train_op,self.loss,self.targets],feed_dict=feed_dict)
            print('train loss = %s ' % loss_value)
            cnt += 1
            if cnt % 10 == 0:
                valid_loss = self.valid()    
                print('train loss = %s, valid loss = %s' %(loss_value, valid_loss))
                path_res = '%s/res' % self.path_prefix
                if not os.path.exists(path_res):
                    os.makedirs(path_res)
                logits_value.tofile('%s/logits.train.bin' % path_res)
                label.tofile('%s/label.train.bin' % path_res)
    
    def valid(self):
        valid_batch_size, feature_x, feature_y, label, pre_label = self.dp.valid_batch()
        
        # train_lstm_c_in,train_lstm_h_in = self.get_init_lstm_state(valid_batch_size)
        feed_dict = {
                self.x_input: feature_x,
                self.y_input: feature_y,
                self.labels: np.expand_dims(label, -1),
                self.pre_label:  pre_label,
                self.batch_size: feature_x.shape[0]
            }
        loss_value,logits_value = self.sess.run([self.loss,self.targets],feed_dict=feed_dict)
        # print(feature_x.shape,feature_y.shape, label.shape, pre_label.shape, logits_value.shape)
        # print('valid_loss = 'loss_value)
        
        path_res = '%s/res' % self.path_prefix
        if not os.path.exists(path_res):
            os.makedirs(path_res)
        logits_value.tofile('%s/logits.bin' % path_res)
        label.tofile('%s/label.bin' % path_res)
        return loss_value

    

def unit_test():
    sys.path.append(os.path.join(os.path.dirname(__file__), '../expriment'))
    from dataset_monitor import dataset_monitor_value_period
    df, x_cols, y_cols, label_cols, label_restore_names = dataset_monitor_value_period()

    _date = ['2017-10-30 00:00:00','2017-11-18 00:00:00','2017-11-18 00:00:00','2017-11-19 00:00:00','2017-11-18 00:00:00','2017-11-19 00:00:00']
    dp = TimeWindowDataPrepare(df, x_cols, y_cols, label_cols[0], _date)
    path_prefix = '../res/multi_step_lstm/'
    
    model = EncoderDecoderModel(dp, path_prefix)
    model.init_network()
    model.train()


    
if __name__ == '__main__':
    unit_test()



        
    
