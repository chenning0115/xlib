# @author charnix@pku.edu.cn

import os,sys
sys.path.append(os.path.dirname(__file__))
sys.path.append(os.path.join(os.path.dirname(__file__),'..'))
import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow.contrib.rnn as rnn

from base_trainer import *

class BatchPeriodSampling(object):
    """
    生成perid的batch训练数据，注意，尚不支持多监测点
    """
    def __init__(self, x_train, y_train, x_valid, y_valid, x_test, y_test):
        """
        parameters:
        -----------
        x_train, x_valid, x_test: pd.DataFrame
        y_trian, y_valid, y_test: pd.Series

        self.* : ndarray
        """
        self.x_train = x_train.values
        self.x_valid = x_valid.values
        self.x_test = x_test.values
        self.y_train = y_train.values.reshape(-1,1)
        self.y_valid = y_valid.values.reshape(-1,1)
        self.y_test = y_test.values.reshape(-1,1)

        
    def train_batch(self, batch_size=64, step_size=20, is_epoch=True, epoch_num=5):
        """train_batch
        1.random shuffle list[0,train.size-1]
        2.random select batch_size instance as one batch
        3.if is_epoch, one epoch is all instance has been selected. there is no index in [0,train.size-1] not been selected
        注意：
        1) 有放回采样与无放回采样，每个epoch中包含的batch_num计算方法一致，epoch_num约束了该方法返回的最大样本数目
        """

        indexes = np.arange(self.x_train.shape[0] - step_size)
        shuffled_indexes = indexes[:]
        np.random.shuffle(shuffled_indexes)
        batch_num_per_epoch = len(indexes) // batch_size
        batch_num = epoch_num * batch_num_per_epoch
        shuffled_cur_i = 0
        for _ in range(batch_num):
            if is_epoch:
                shuffled_cur_i = 0 if shuffled_cur_i+batch_size > len(indexes) else shuffled_cur_i
                select_indexes = shuffled_indexes[shuffled_cur_i:shuffled_cur_i+batch_size]
                shuffled_cur_i += batch_size
            else:
                select_indexes = np.random.choice(indexes,batch_size,replace=False)
            
            batch_data_x = []
            batch_data_y = []
            for i in select_indexes:
                batch_data_x.append(self.x_train[i:i+step_size,:])
                batch_data_y.append(self.y_train[i:i+step_size,:])
            
            yield np.stack(batch_data_x),np.stack(batch_data_y)

    def valid_test_batch(self, x_data, y_data):
        """
        x_data: [step, feature] --> [1, step, feature]
        y_data:[step, 1] --> [1, step, 1]
        """
        return np.expand_dims(x_data, axis=0), np.expand_dims(y_data, axis=0)



    def valid_batch(self):
        return self.valid_test_batch(self.x_valid, self.y_valid)
    
    def test_batch(self):
        return self.valid_test_batch(self.x_test, self.y_test)



class LSTMModel(object):
    def __init__(self, period=20, feature_num=10, unit_num=8, fc1_num=8, fc2_num=1, lr=0.01, clip_grads_max=8):
        self.period = period
        self.feature_num = feature_num
        self.unit_num = unit_num
        self.fc1_num = fc1_num
        self.fc2_num = fc2_num

        self.lr = lr
        self.clip_grads_max = clip_grads_max
        self.init_network()


    def init_network(self):
        with tf.variable_scope('global_scope') as vs:
            self.logits =  self.inference()
            self.loss = self.get_loss(self.logits)

            global_step = tf.Variable(0,trainable=False)
            lr_decay = self.lr
            self.opt = tf.train.AdamOptimizer(learning_rate=lr_decay)

            self.train_variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, vs.name)
            grads_vars = self.opt.compute_gradients(self.loss, self.train_variables)
            # clip the gradients ensure the gradient not explode
            for i, (g,v) in enumerate(grads_vars):
                if g is not None:
                    grads_vars[i] = (tf.clip_by_norm(g, self.clip_grads_max), v)
            self.train_op = self.opt.apply_gradients(grads_vars,global_step=global_step)

            self.sess = tf.Session()
            init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
            self.sess.run(init_op)
            print('init newwork done.')

    def train(self, x_train, y_train):
        """just train one step(one batch)
        parameters:
        -----------
        x_train: pd.DataFrame or ndarray shape=(batch, step, features)
        y_train: pd.DataFrame or ndarray shape=(batch, step, 1)
        """
        assert x_train.shape[0] == y_train.shape[0]
        batch_size = x_train.shape[0]
        train_lstm_c_in,train_lstm_h_in = self.get_init_lstm_state(batch_size)
        feed_dict = {
            self.input: x_train,
            self.target: y_train,
            self.lstm_c_in : train_lstm_c_in,
            self.lstm_h_in : train_lstm_h_in
        }
        _,loss_value,logits_value = self.sess.run([self.train_op,self.loss,self.logits],feed_dict=feed_dict)
        return loss_value, logits_value

    def valid(self, x_valid, y_valid):
        assert x_valid.shape[0] == y_valid.shape[0]
        batch_size = x_valid.shape[0]
        train_lstm_c_in,train_lstm_h_in = self.get_init_lstm_state(batch_size)
        feed_dict = {
            self.input: x_valid,
            self.target: y_valid,
            self.lstm_c_in : train_lstm_c_in,
            self.lstm_h_in : train_lstm_h_in
        }
        loss_value,logits_value = self.sess.run([self.loss,self.logits],feed_dict=feed_dict)
        return loss_value, logits_value

    def predict(self, x_test):
        batch_size = x_test.shape[0]
        train_lstm_c_in,train_lstm_h_in = self.get_init_lstm_state(batch_size)
        feed_dict = {
            self.input: x_valid,
            self.lstm_c_in : train_lstm_c_in,
            self.lstm_h_in : train_lstm_h_in
        }
        logits_value = self.sess.run([self.logits],feed_dict=feed_dict)
        return logits_value
        
    def get_init_lstm_state(self, batch_size):
        return np.zeros((batch_size, self.lstm_c_size)), np.zeros((batch_size, self.lstm_h_size))

    def _get_fc_variable(self, weight_shape):
        d = 1.0 / np.sqrt(weight_shape[0])
        bias_shape = [weight_shape[1]]
        weight = tf.Variable(tf.truncated_normal(weight_shape, stddev=d,dtype=tf.float32,name='weights'))
        bias = tf.Variable(tf.random_uniform(shape=bias_shape, minval=-d, maxval=d, name='bias',dtype=tf.float32))
        tf.add_to_collection('losses',tf.nn.l2_loss(weight))
        return weight, bias

    def inference(self):
        # [batch,period,feature]
        self.input = tf.placeholder(tf.float32, shape=(None, None, self.feature_num))

        lstm_cell = rnn.BasicLSTMCell(self.unit_num, state_is_tuple=True)
        self.lstm_c_size = lstm_cell.state_size.c
        self.lstm_h_size = lstm_cell.state_size.h

        # [batch_size, lstm_c_size or lstm_h_size]
        #  where lstm_c_size or lstm_h_size is always lstm_num_units
        self.lstm_c_in = tf.placeholder(tf.float32, shape=(None, self.lstm_c_size))
        self.lstm_h_in = tf.placeholder(tf.float32, shape=(None, self.lstm_h_size))

        with tf.variable_scope('scope_lstm') as vs_lstm:
            lstm_state_in = rnn.LSTMStateTuple(self.lstm_c_in, self.lstm_h_in)
            # lstm_output:[batch,step,units]
            # lstm_state:state_tuple
            lstm_output, lstm_state = tf.nn.dynamic_rnn(
                lstm_cell,
                self.input,
                initial_state=lstm_state_in,
                time_major=False
            )
            # lstm_fc_output: [batch*step, unit]
            lstm_fc_output = tf.reshape(lstm_output, (-1,self.unit_num))
            # w1:[unit, fc1]
            # b1:[fc1]
            fc_w1, fc_b1 = self._get_fc_variable([self.unit_num, self.fc1_num])
            fc1_output = tf.layers.batch_normalization(tf.matmul(lstm_fc_output, fc_w1) + fc_b1)
            fc1_output = tf.nn.relu(fc1_output)

            fc_w2, fc_b2 = self._get_fc_variable([self.fc1_num, self.fc2_num])
            fc2_output = tf.layers.batch_normalization(tf.matmul(fc1_output, fc_w2) + fc_b2)
            fc2_output = tf.nn.relu(fc2_output)
            
        return fc2_output

    def get_loss(self, logits):
        # target:[batch,period,1]
        self.target = tf.placeholder(tf.float32, shape=(None, None, 1))
        # temp_target: [batch*period, 1]
        target_reshape = tf.reshape(self.target, (-1,1))

        loss = tf.reduce_mean(tf.pow(logits - target_reshape, 2))
        tf.add_to_collection('losses', loss)

        return loss

    

"""
注意注意注意：
输入的数据必须不能有nan！！！
"""
class LSTMTrainer(BaseTrainer):
    """LSTM trainer
    one step
    """
    def __init__(self, _dp, _path_prefix):
        super().__init__(_dp, _path_prefix)
        # train parameters
        self.batch_size = 64
        self.step_size = 100
        self.is_epoch = True
        self.epoch_num = 20

        # model parameters
        self.feature_num=25
        self.unit_num=16
        self.fc1_num=16
        self.fc2_num=1
        self.lr=0.02
        self.clip_grads_max=8

        self.bps = None

    def train(self, train_data, valid_data, test_data):
        self.model = LSTMModel(period=self.step_size, feature_num=self.feature_num, 
            unit_num=self.unit_num, fc1_num=self.fc1_num, fc2_num=self.fc2_num, 
            lr=self.lr, clip_grads_max=self.clip_grads_max)

        x_train, y_train = self.split_xy(train_data)
        x_valid, y_valid = self.split_xy(valid_data)
        x_test, y_test = self.split_xy(test_data)
        self.bps = BatchPeriodSampling(x_train, y_train, x_valid, y_valid, x_test, y_test)

        x_valid_batch, y_valid_batch = self.bps.valid_batch()
        x_test_batch, y_test_batch = self.bps.test_batch()

        batch_i = 0
        for x_batch_train, y_batch_train in self.bps.train_batch( \
            batch_size=self.batch_size,step_size=self.step_size, \
            is_epoch=self.is_epoch, epoch_num=self.epoch_num):
            # print('data shape is ', x_batch_train.shape, y_batch_train.shape)
            train_loss, train_logits = self.model.train(x_batch_train, y_batch_train)
            print('train %d: loss=%s' % (batch_i, train_loss))
            if batch_i % 10 == 0:
                valid_loss, valid_logits = self.model.valid(x_valid_batch, y_valid_batch)
                print('== valid %d: loss=%s' % (batch_i, valid_loss))
            batch_i += 1

        print('train done!')
        return self.model

    def predict(self, test_data):
        assert self.bps is not None
        x_test, y_test = self.split_xy(test_data)
        x_test_batch, y_test_batch = self.bps.test_batch()
        test_loss, test_logits = self.model.valid(x_test_batch, y_test_batch)
        print('test_logist_shape is : ',test_logits.shape)
        df_res = pd.DataFrame()
        df_res['ori'] = y_test
        df_res['pred'] = test_logits

        path_pred = '%s/pred.csv' % self.path_prefix
        df_res.to_csv(path_pred)
        return df_res


    def evaluate(self, pred):
        pass
    


if __name__ == "__main__":
    from expriment.dataset_monitor import gen_monitor_dp
    dp = gen_monitor_dp()
    for train ,valid , test in dp.next():
        bp = BatchPeriodSampling(train, valid, test)
        for batch in bp.train_batch():
            print(batch.shape)