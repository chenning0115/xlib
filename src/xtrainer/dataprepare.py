# @author charnix@pku.edu.cn

import os,sys
sys.path.append(os.path.dirname(__file__))

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
    
    def next(self):
        for date_tuple in self.dates:  #支持滚动训练
            train_start, train_end, valid_start, valid_end, test_start, test_end \
                = [pd.Timestamp(d) for d in date_tuple]
            train_data = self.data.loc(axis=0)[:, train_start:train_end]
            valid_data = self.data.loc(axis=0)[:, valid_start:valid_end]
            test_data = self.data.loc(axis=0)[:, test_start:test_end]
            yield train_data, valid_data, test_data

        

def unit_test():
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
    from dataset_monitor import dataset_monitor_value
    df, feature_cols, label_cols = dataset_monitor_value()

    _dates = [['2017-11-01 00:00:00','2017-11-27 00:00:00','2017-11-27 00:00:00','2017-11-29 00:00:00','2017-11-29 00:00:00','2017-11-30 00:00:00']]
    dp = DataPrepare(df, feature_cols, label_cols[0], _dates)
    for tup in dp.next():
        print(len(tup))
        train, valid, test = tup
        return train,valid,test

    
if __name__ == '__main__':
    unit_test()
        
    
        
        

    



