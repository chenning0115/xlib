# @author charnix@pku.edu.cn

import os,sys
sys.path.append(os.path.dirname(__file__))
sys.path.append(os.path.join(os.path.dirname(__file__),'..'))
from xdata.base import eval_dataset
from collections import OrderedDict
import pandas as pd


def dataset_monitor_value():
    # 特征数据
    names = []
    features = []
    # 当前天的监测值
    features += ["$('MonitorValue')"]
    names += ['raw_value']

    # 历史均值、 最大值、 最小值、 标准差、 峰度、 偏度
    for ts in ['RMean', 'RMax', 'RMin', 'RStd', 'RKurt', 'RSkew']:
        features += ["%s($('MonitorValue'),%s)" % (ts,i) for i in [5,15,30,60]]
        names += ["%s_%s" % (ts,i) for i in [5,15,30,60]]

    # label数据
    label_names = ['LABEL0']
    labels = ["Ref($('MonitorValue'),-1)"]

    feature_dict = OrderedDict({})
    for n,f in zip(names, features):
        feature_dict[n] = f
    for n,f in zip(label_names, labels):
        feature_dict[n] = f
    df = eval_dataset(feature_dict) 
    return df, names, label_names


    
def gen_monitor_dp():
    from xtrainer.dataprepare import DataPrepare
    df, feature_cols, label_cols = dataset_monitor_value()

    _dates = [
        ['2017-11-01 00:00:00','2017-11-27 00:00:00',
        '2017-11-27 00:00:00','2017-11-29 00:00:00',
        '2017-11-29 00:00:00','2017-11-30 00:00:00']
                                                    ]
    dp = DataPrepare(df, feature_cols, label_cols[0], _dates, _fill_nan=True)
    return dp

if __name__ == '__main__':
    
    from xtrainer.base_trainer import LgbRegTrainer
    from xtrainer.lstm_trainer import LSTMTrainer
    dp = gen_monitor_dp()
    dp.data.to_pickle('../../data/feature_monitor.pkl')
    path_prefix = '../res/lstm_one_step_test'
    if not os.path.exists(path_prefix):
        os.makedirs(path_prefix)
    lgb_trainer = LSTMTrainer(dp, path_prefix)
    lgb_trainer.run()