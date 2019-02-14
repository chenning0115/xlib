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

    #前N个时间节点的绝对值
    features += ["Ref($('MonitorValue'),%s)" % i for i in range(1,15)]
    # 历史均值、 最大值、 最小值、 标准差、 峰度、 偏度
    # for ts in ['RMean', 'RMax', 'RMin', 'RStd', 'RKurt', 'RSkew']:
    #     features += ["%s($('MonitorValue'),%s)" % (ts,i) for i in [15,20,25,30,35,40,45,50,55,60]]
    #     names += ["%s_%s" % (ts,i) for i in [15,20,25,30,35,40,45,50,55,60]]

    # label数据
    label_names = ['LABEL0']
    labels = ["Ref($('MonitorValue'),-1)"]
    # labels = ["$('MonitorValue')"]

    feature_dict = OrderedDict({})
    for n,f in zip(names, features):
        feature_dict[n] = f
    for n,f in zip(label_names, labels):
        feature_dict[n] = f
    df = eval_dataset(feature_dict) 
    return df, names, label_names


# ['009A13', '044A02', '044A03', '044A09', '044A11', '044A15']
# 009A13：工作面进风甲烷
# 044A02：上隅角甲烷
# 044A03：工作面回风甲烷
# 044A09：粉尘
# 044A11：回风混合甲烷
# 044A15：回风混合风速
#用于多步预测
#test_window=30 1个小时的时间预测
def dataset_monitor_value_period():
    # 特征数据
    x_names = []
    features = []
    # 当前天的监测值
    for c in ['009A13','044A02','044A09']:
        #当前时间节点的值
        features += ["$('%s')" % c]
        x_names += ['raw_value_%s' % c]

        # #前N个时间节点的绝对值
        # l = 5
        # features += ["Ref($('%s'),%s)" % (c,i) for i in range(1,l)]
        # x_names += ["previous_value_%s_%s" % (c,i) for i in range(1,l)]

        # 当前时间节点与之前时间节点的差值
        kr = 5
        features += ["$('%s') - Ref($('%s'),%s)" % (c,c,i) for i in range(1, kr)]
        x_names += ["previous_diff_value_%s_%s" % (c,i) for i in range(1, kr)]

        # 滚动差值
        k = 5
        features += ["Ref($('%s'),%s) - Ref($('%s'),%s)" % (c,i,c,i+1) for i in range(k)]
        x_names += ["previous_rolling_diff_value_%s_%s" % (c,i) for i in range(k)]

    # # 测试是否引入未来信息
    # features += ["Ref($('044A02'),-1)"]
    # x_names += ["test_label"]

    # 历史均值、 最大值、 最小值、 标准差、 峰度、 偏度
    # for ts in ['RMean','RStd']:
    #     features += ["%s($('MonitorValue'),%s)" % (ts,i) for i in [10,15,20,25,30]]
    #     x_names += ["%s_%s" % (ts,i) for i in [10,15,20,25,30]]

    # 不同传感器间差值
    # target_sensor = '044A02'
    # diff_sensor = ['009A13']
    # for s in diff_sensor:
    #     features += ["$('%s') - Ref($('%s'),%s)" % (target_sensor, s, 1)]
    #     x_names += ["diff(%s,(%s,%s))" % (target_sensor, s, 1)]

    # 开始准备y的features

    y_names = []

    names = x_names + y_names
    # label数据
    label_names = ['LABEL0']
    labels = ["Ref($('044A02'),-1) - $('044A02')"]

    label_restore_names = ['LABEL_STORE']
    label_restore = ["Ref($('044A02'),-1) - $('044A02')"]
    # labels = ["$('MonitorValue')"]

    feature_dict = OrderedDict({})
    for n,f in zip(names, features):
        feature_dict[n] = f
    for n,f in zip(label_names, labels):
        feature_dict[n] = f
    for n,f in zip(label_restore_names, label_restore):
        feature_dict[n] = f
    df = eval_dataset(feature_dict) 

    return df, x_names,y_names,label_names,label_restore_names


    
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
    print('load data done. shape=%s' % str(dp.data.shape))
    # dp.data.to_pickle('../../data/feature_monitor.pkl')
    path_prefix = '../res/lstm_feature_1'
    if not os.path.exists(path_prefix):
        os.makedirs(path_prefix)
    model_args = {
        'batch_size' : 256,
        'step_size'  : 30,
        'is_epoch' : True,
        'epoch_num' : 5,
        'feature_num' : dp.get_final_feature_num(),
        'unit_num' : 16,
        'fc1_num' : 16,
        'fc2_num' : 1,
        'lr':0.02,
        'clip_grads_max':8
    }
    lgb_trainer = LSTMTrainer(dp, path_prefix, **model_args)
    lgb_trainer.run()