# @author charnix@pku.edu.cn
from base import eval_dataset
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


    

if __name__ == '__main__':
    df = dataset_monitor_value() 
    print(df.shape)
    print(df.columns)
    df = df[~pd.isnull(df['LABEL0'])]
    print(df.shape)
    print(df.iloc[-3:])