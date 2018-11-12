# @author charnix@pku.edu.cn


import lightgbm as lgb
import pandas as pd

from dataprepare import DataPrepare
from dataset_monitor import dataset_monitor_value
from sklearn.metrics import mean_squared_error

import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

class BaseTrainer(object):
    def __init__(self, _dp, _path_prefix):
        """BaseTrainer
        Paremeters:
        -----------
        _dp : DataPrepare 
            DataPrepare Instance, prepare data
        """
        self.dp = _dp
        self.path_prefix = _path_prefix
        self.model = None

    def run(self):
        """run
        run train and predict and final evaluate 
        get the result
        """
        for train_data, valid_data, test_data in self.dp.next():
            self.model = self.train(train_data, valid_data, test_data)
            pred = self.predict(test_data)
            res = self.evaluate(pred)
            print('not Implement now.')

    def split_xy(self, part_data):
        return part_data[self.dp.final_features], part_data[self.dp.final_label]

    def train(self, train_data, valid_data, test_data):
        """train model
        Parameters:
        -----------
        train_data: pd.DataFrame
        valid_data: pd.DataFrame
        test_data: pd.DataFrame

        Returns:
        -----------
        model: real model(lgb, nn, etc.)
        """
        pass

    def predict(self, test_data):
        pass

    def evaluate(self, pred):
        pass



class LgbRegTrainer(BaseTrainer):
    def __init__(self, _dp, _path_prefix):
        super().__init__(_dp, _path_prefix)

    def train(self, train_data, valid_data, test_data):
        x_train, y_train = self.split_xy(train_data)
        x_valid, y_valid = self.split_xy(valid_data)
        x_test, y_test = self.split_xy(test_data)

        lgb_train = lgb.Dataset(x_train, y_train)
        lgb_eval = lgb.Dataset(x_valid, y_valid, reference=lgb_train)
        params = {
            'boosting_type': 'gbdt',
            'objective': 'regression',
            'metric': {'l2', 'l1'},
            'num_leaves': 31,
            'learning_rate': 0.05,
            'feature_fraction': 0.9,
            'bagging_fraction': 0.8,
            'bagging_freq': 5,
            'verbose': 0
        }
        model = lgb.train(params,
                lgb_train,
                num_boost_round=200,
                valid_sets=lgb_eval,
                early_stopping_rounds=5)

        return model

    def predict(self, test_data):
        x_test, y_test = self.split_xy(test_data)
        y_pred = self.model.predict(x_test, num_iteration=self.model.best_iteration)
        df_res = pd.DataFrame()
        df_res['ori'] = y_test
        df_res['pred'] = y_pred

        path_pred = '%s/pred.csv' % self.path_prefix
        df_res.to_csv(path_pred)
        return df_res

    def evaluate(self, pred):
        pass
    


if __name__ == '__main__':
    df, feature_cols, label_cols = dataset_monitor_value()

    _dates = [
        ['2017-11-01 00:00:00','2017-11-27 00:00:00',
        '2017-11-27 00:00:00','2017-11-29 00:00:00',
        '2017-11-29 00:00:00','2017-11-30 00:00:00']
                                                    ]
    dp = DataPrepare(df, feature_cols, label_cols[0], _dates)
    path_prefix = '../res/lgb_reg_test'
    if not os.path.exists(path_prefix):
        os.makedirs(path_prefix)
    lgb_trainer = LgbRegTrainer(dp, path_prefix)
    lgb_trainer.run()
