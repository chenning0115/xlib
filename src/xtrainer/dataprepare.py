# @author charnix@pku.edu.cn

import os,sys
sys.path.append(os.path.dirname(__file__))

import numpy as np
import pandas as pd


class DataPrepare(object):
    def __init__(self, _df_data, _ori_features, _ori_label, _dates, _fill_nan=False):
        self.df_data = _df_data
        self.ori_features = _ori_features # not include any labels
        self.ori_label = _ori_label # just one label 

        self.final_features = None
        self.final_label = None

        self.dates = _dates
        # print(self.dates)

        self.fill_nan = _fill_nan

        self.data = self.prepare(self.df_data)

    def prepare(self, data):
        # prepare feature data like normalization
        # pass

        self.final_features = self.ori_features
        # prepare label
        self.final_label = self.ori_label

        # fill nan or not
        if self.fill_nan:
            data = data.fillna(method='ffill')

        data = data[~pd.isnull(data[self.final_label])]


        cols = self.final_features + [self.final_label] #confirm last column is the final label and other cols not inclue any labels.
        data = data[cols]
        return data

    def get_final_feature_num(self):
        return len(self.final_features)

    def next(self):
        for date_tuple in self.dates:  #支持滚动训练
            train_start, train_end, valid_start, valid_end, test_start, test_end \
                = [pd.Timestamp(d) for d in date_tuple]
            train_data = self.data.loc(axis=0)[:, train_start:train_end]
            valid_data = self.data.loc(axis=0)[:, valid_start:valid_end]
            test_data = self.data.loc(axis=0)[:, test_start:test_end]
            yield train_data, valid_data, test_data

        


class  TimeWindowDataPrepare(object):
    def __init__(self, _df_data, _ori_x_features, _ori_y_features, _ori_label, _date,
                        batch_size=1024, train_window = 5, test_window=1, raw_label_input_size=1, epoch_num=50,
                        normalize_features = True
        ):
        self.df_data = _df_data
        self.ori_x_features = _ori_x_features # not include any labels
        self.ori_y_features = _ori_y_features
        self.ori_label = _ori_label # just one label 

        # batch data
        self.batch_size = batch_size
        self.train_window = train_window
        self.test_window = test_window
        self.raw_label_input_size = raw_label_input_size

        # 注意，step_size这里之所以会减去1是因为df的每一行数据中label为该行特征作为输入时，需要预测的label，
        # 因此多步预测时，真正的test_window与train_window有一个重合步
        self.step_size = train_window + test_window - 1 

        self.epoch_num = epoch_num

        self.final_features = None
        self.final_x_features = None
        self.final_y_features = None
        self.final_label = None

        self.date = _date
        # print(self.dates)

        self.fill_nan = True #此种模式下必须强制fill nan，保证时间序列的连续性
        self.normalize_features = normalize_features # 默认需要进行列的标准化
        self.data = self.prepare(self.df_data)

        self.train_data, self.valid_data, self.test_data = self.__split_date(self.data)
        self.valid_batch_data = None
        self.test_batch_data = None

    def __split_date(self, data):
        train_start, train_end, valid_start, valid_end, test_start, test_end \
                = [pd.Timestamp(d) for d in self.date]
        train_data = data.loc(axis=0)[:, train_start:train_end]
        valid_data = data.loc(axis=0)[:, valid_start:valid_end]
        test_data = data.loc(axis=0)[:, test_start:test_end]
        return train_data, valid_data, test_data


    def prepare(self, data):
        # prepare feature data like normalization
        # pass

        self.final_x_features = self.ori_x_features
        self.final_y_features = self.ori_y_features
        # prepare label
        self.final_label = self.ori_label

        # fill nan or not
        if self.fill_nan:
            data = data.fillna(method='ffill')

        data = data[~pd.isnull(data[self.final_label])]

        self.final_features = self.final_x_features + self.final_y_features

        if self.normalize_features:
            for fea in self.final_features:
                data[fea] = (data[fea] - data[fea].mean())/data[fea].std()

        cols = self.final_features + [self.final_label] #confirm last column is the final label and other cols not inclue any labels.
        data = data[cols]
        return data

    def get_final_feature_num(self):
        return len(self.final_features)

    def get_x_feature_num(self):
        return len(self.final_x_features)

    def get_y_feature_num(self):
        return len(self.final_y_features)


    def generate_batch(self, data, batch_size, step_size, train_window, test_window, 
                        epoch_num, is_epoch=True):
        """train_batch
        1.random shuffle list[0,train.size-1]
        2.random select batch_size instance as one batch
        3.if is_epoch, one epoch is all instance has been selected. there is no index in [0,train.size-1] not been selected
        注意：
        1) 有放回采样与无放回采样，每个epoch中包含的batch_num计算方法一致，epoch_num约束了该方法返回的最大样本数目

        Parameters:
        -----------
        data: pd.DataFrame
        batch_size: int  for batch size
        step_size, train_window, test_window: int assert train + test - 1 == step
        epoch_num: int for epoch num

        Returns:
        ----------
        x_features: ndarray [batch, step, x_feature_size]
        y_features: ndarray [batch, step, y_features_size]
        label: ndarray [batch, step, 1]

        """
        assert step_size == (train_window + test_window - 1)
        indexes = np.arange(data.shape[0] - step_size)
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

            # 开始按照生成的select_indexes来进行切分
            data_x_features = data[self.final_x_features]
            # print('data_x_features',len(self.final_x_features),data_x_features.shape)
            data_y_features = data[self.final_y_features]
            data_label = data[self.final_label]
            batch_data_x_features = []
            batch_data_y_features = []
            batch_data_label = []
            batch_pre_label = []
            for i in select_indexes:
                batch_data_x_features.append(data_x_features.iloc[i:i+train_window,:].values)
                batch_data_y_features.append(data_y_features.iloc[i+train_window-1:i+step_size,:].values) #i+train_window-1是由于data中的train_window,test_window间会有一个重合步
                batch_data_label.append(data_label.iloc[i+train_window-1:i+step_size].values)
                batch_pre_label.append(data_label.iloc[i+train_window-1-self.raw_label_input_size:
                                                        i+train_window-1].values) #i+train_window-1位置的值对应于本次预测的第一个label
            
            yield (np.stack(batch_data_x_features),
                  np.stack(batch_data_y_features),
                  np.stack(batch_data_label),
                  np.stack(batch_pre_label)) # return shape [batch, step, all_features]


    def train_batch(self):
        for data_tuple in self.generate_batch(self.train_data, self.batch_size, 
            self.step_size, self.train_window, self.test_window, self.epoch_num):
            feature_x, feature_y, label, pre_label = data_tuple
            yield feature_x, feature_y, label, pre_label
        

    def generate_valid_test_batch(self, data, step_size, train_window, test_window):
        assert train_window + test_window - 1 == step_size
        indexes = np.arange(data.shape[0] - step_size) # 注意这里使得少了最后一个预测的值
        batch_size = len(indexes) 
        # 开始按照生成的select_indexes来进行切分
        data_x_features = data[self.final_x_features]
        # print('data_x_features',len(self.final_x_features),data_x_features.shape)
        data_y_features = data[self.final_y_features]
        data_label = data[self.final_label]
        batch_data_x_features = []
        batch_data_y_features = []
        batch_data_label = []
        batch_pre_label = []
        for i in indexes:
            batch_data_x_features.append(data_x_features.iloc[i:i+train_window,:].values)
            batch_data_y_features.append(data_y_features.iloc[i+train_window-1:i+step_size,:].values)
            batch_data_label.append(data_label.iloc[i+train_window-1:i+step_size].values)
            batch_pre_label.append(data_label.iloc[i+train_window-1-self.raw_label_input_size:
                                                        i+train_window-1].values)
        return (batch_size, np.stack(batch_data_x_features),
                np.stack(batch_data_y_features),
                np.stack(batch_data_label),
                np.stack(batch_pre_label)) # return shape [batch, step, all_features]

    def valid_batch(self):
        if self.valid_batch_data is None:
            self.valid_batch_data = self.generate_valid_test_batch(self.valid_data, step_size = self.step_size,
                                            train_window = self.train_window, 
                                            test_window = self.test_window)
        return self.valid_batch_data

    def test_batch(self):
        if self.test_batch_data is None:
            self.test_batch_data = self.generate_valid_test_batch(self.test_data, step_size = self.step_size,
                                            train_window = self.train_window, 
                                            test_window = self.test_window)
        return self.test_batch_data


def unit_test():
    sys.path.append(os.path.join(os.path.dirname(__file__), '../expriment'))
    from dataset_monitor import dataset_monitor_value_period
    df, x_cols, y_cols, label_cols = dataset_monitor_value_period()

    _date = ['2017-11-27 00:00:00','2017-11-28 00:00:00','2017-11-29 00:00:00','2017-11-30 00:00:00','2017-11-29 00:00:00','2017-11-30 00:00:00']
    dp = TimeWindowDataPrepare(df, x_cols, y_cols, label_cols[0], _date)
    dp.train_batch()

    
if __name__ == '__main__':
    unit_test()
        
    
        
        

    



