# @author charnix@pku.edu.cn

import os,sys
sys.path.append(os.path.dirname(__file__))

import lightgbm as lgb
import pandas as pd

from dataprepare import DataPrepare
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
    """LgbRegTrainer
    Just for example.
    """
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
    



