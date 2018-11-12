# xlib
one project for training time-series data prediction model automatically.


### 本项目为建立时间序列预测模型提供框架支持，具体包括
####  基于表达式的自动特征构建
```
n2expr = {
        'mv':"$('MonitorValue')",  
        'diff':"Ref($('MonitorValue'),-1) - $('MonitorValue')", 
        'mv-mean_mv':"$('MonitorValue') - RMean($('MonitorValue'),2)",
        }
name2expr = OrderedDict(n2expr)
```
利用表达式来描述特征的处理过程，从而快速构建特征，上述三个表达式分别表示
1. 将原始数据中列为MonitorValue的数据作为一个特征
2. 表示对MonitorValue列先进行上移一个sample位置并减去当前位置值，相当于计算一阶残差
3. 表示当前值减去前两个时间点的移动均值

```
dataset = DataSet(name2expr)
df = dataset.trans()
```
建立一个DataSet类对象，传入表达式字典，并调用trans()方法， 从而将表达式转化为对应的pandas DataFrame对象，该对象为multi-index(id, Timestamp)

#### 当前支持操作符
feature 为Feature类的对象，f为数字或Feature类对象

```
Ref(feature,n)
Abs(feature)
f1 + f2 
f1 - f2
f1 * f2
f1 / f2
Gt(feature1,feature2)
Rmax(feature,n)
RMin(feature,n)
RMedian(feature,n)
Rmean(feature,n)
RQuant(feature,n)
RKurt(feature,n)
RSkew(feature,n)
RSum(feature,n)
RVar(feature,n)
RStd(feature,n)
Resi(feature,n)
EWMA(feature,n)

```
####  数据切分与模型训练

```
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
```
