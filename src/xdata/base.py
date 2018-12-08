# @author charnix@pku.edu.cn

import os,sys
sys.path.append(os.path.dirname(__file__))

import numpy as np
import pandas as pd
from collections import OrderedDict

import config
from data_loader import xdata_loader
from cache import xcache

class Feature(object):
    """Feature
    Base Feature Class

    Attribute:
    ---------
    data: pandas Series
    data must have multi-index, for example:
        (id, datetime) where
        'id' represents one time-series data
        'datetime' represents timestamp
        note: data has been sorted by multi-index(id, datetime)
    
    Function:
    ---------
    load_feature

    """
    def __init__(self):
        self.data = None  # Series object that may have multi-index(id, datetime)

    def __str__(self):
        return type(self).__name__
    
    def load_feature(self):
        """load_feature
        load specific feature, set Serise data to self.data
        """
        # first check cache, if feature has been loaded, just get from cache.
        key_cache = str(self)
        if xcache.check(key_cache):
            self.data = xcache.get(key_cache)
            # print('cache, find %s' % key_cache)
        # other-wise load by specific funciton 'load_internal'
        else:
            temp_data = self.load_internal()
            xcache.put(key_cache, temp_data)
            # print('cache', key_cache, temp_data.iloc[-1])
            self.data = temp_data
            # print('cache, not find %s' % key_cache)

    def load_internal(self):
        """load_internal
        specific load feature function, sub-class should implement
        
        return
        -------
        Pandas Series
        should return series data that will be set to self.data
        """

        pass


    # TODO: rewrite internal expression
    def __add__(self, other):
        return Add(self, other)
    def __radd__(self, other):
        return Add(other, self)

    def __sub__(self, other):
        return Sub(self, other)
    def __rsub__(self, other):
        return Sub(other, self)

    def __mul__(self, other):
        return Mul(self, other)
    def __rmul__(self, other):
        return Mul(other, self)

    def __truediv__(self, other):
        return Div(self, other)
    def __rtruediv__(self, other):
        return Div(other, self)

    

class StaticFeature(Feature):
    """StaticFeature
    The Feature load from disk.
    """
    def __init__(self, _field_name):
        self.field_name = _field_name
    
    def __str__(self):
        return '%s(%s)' % (type(self).__name__, self.field_name)


    def load_internal(self):
        """
        load by dataloader from disk
        """
        temp_data = xdata_loader.load_data_static_feature(self.field_name)
        assert temp_data is not None
        return temp_data
        
class OpsFeature(Feature):
    """OpsFeature
    The Feature generate by other Features(can be StaticFeature or OpsFeature)
    """

    def __init__(self):
        super().__init__()

    def __str__(self):
        return '%s' % (type(self).__name__)

    def load_internal(self):
        pass


# TODO: Be Careful: If you want to use ref, you must confirm that index-Time must be uniform. one by one
class Ref(OpsFeature):
    """Ref Feature
    Move sample to next N datetime-stamp for each id.
    parameters:
    ------------
    feature: Feature
        input feature

    n : integer
        n step

    Returuns:
    -----------
    Feature: Feature
        return feature instance
    """
    def __init__(self, feature, n):
        self.feature = feature
        self.n = n
        super().__init__()

    def __str__(self):
        return '%s(%s,%s)' % (type(self).__name__, str(self.feature), self.n)

    def load_internal(self):
        assert isinstance(self.feature, Feature)
        self.feature.load_feature()
        feature_data = self.feature.data
        return feature_data.groupby(level=0).shift(periods=self.n)


######################################################
#-------------------Element-wise---------------------#
######################################################


class ElemOperator(OpsFeature):
    """Element-wise Operator

    Parameters
    ----------
    feature: Feature
        feature instance

    Returns
    ---------
    Feature
        feature instance output
    """
    def __init__(self, feature, func):
        self.feature = feature
        self.func = func

    def __str__(self):
        return '%s(%s)' %(type(self).__name__ ,str(self.feature))

    def load_internal(self):
        feature_data = None
        if isinstance(self.feature, Feature):
            self.feature.load_feature()
            feature_data = self.feature.data
        else:
            feature_data = self.feature
        return getattr(np, self.func)(feature_data)

class Abs(ElemOperator):
    """Abs
        abs(x)
    """
    def __init__(self, feature, func='abs'):
        super().__init__(feature,func)
    
        
######################################################
#----------------------Pair-wise---------------------#
######################################################

class PairWiseOperator(OpsFeature):
    """PairWiseOperator

    Parameters
    ----------
    feature_left: Feature
        feature instance or other(int, float, etc.)
    feature_right: Feature
        feature instance or other(int, float, ect.)

    Returns
    ---------
    Feature
        feature instance output
    """

    def __init__(self, _feature_left, _feature_right, _func):
        self.feature_left = _feature_left
        self.feature_right = _feature_right
        self.func = _func

    def __str__(self):
        return '%s(%s,%s)' %(type(self).__name__ ,str(self.feature_left), str(self.feature_right))

    def load_internal(self):
        left_data, right_data = None, None
        if isinstance(self.feature_left, Feature):
            self.feature_left.load_feature()
            left_data = self.feature_left.data
        else:
            left_data = self.feature_left
        if isinstance(self.feature_right, Feature):
            self.feature_right.load_feature()
            right_data = self.feature_right.data
        else:
            right_data = self.feature_right
        return getattr(np, self.func)(left_data, right_data)


class Add(PairWiseOperator):
    """Add +
    """
    def __init__(self, _feature_left, _feature_right, _func='add'):
        super().__init__(_feature_left, _feature_right, _func)

class Sub(PairWiseOperator):
    """Substract -
    """
    def __init__(self, _feature_left, _feature_right, _func='subtract'):
        super().__init__(_feature_left, _feature_right, _func)

class Mul(PairWiseOperator):
    """Multiply *
    """
    def __init__(self, _feature_left, _feature_right, _func='multiply'):
        super().__init__(_feature_left, _feature_right, _func)


class Div(PairWiseOperator):
    """Divide /
    """
    def __init__(self, _feature_left, _feature_right, _func='divide'):
        super().__init__(_feature_left, _feature_right, _func)


class Gt(PairWiseOperator):
    """Greater
     left greater than right then true else false
    """
    def __init__(self, _feature_left, _feature_right, _func='greater'):
        super().__init__(_feature_left, _feature_right)
    

######################################################
#--------------------Rolling Feature-----------------#
######################################################

class RollingFeature(OpsFeature):
    """RollingFeature
    generate feature by rolling time-series

    note: input feature has multi-index(id, datetime), 
    one id represents one time-series and one datetime represents one timestamp
    when calculate time-series rolling feature, feature.data will be group by id
    and then call rolling function as well as specific statistic function. 
    for example: data (pandas series, mulitindex(id, datetime))
        data.groupby(level=0).rolling(window).max()
    """
    def __init__(self, _feature, _win, _func, _min_periods=1, **kvargs):

        """
        Parameters
        ----------
        _feature: Feature
            Feature instance
        _win: int
            rolling window size
        _func: str
            specific function or description
        _min_periods: int
            minimum periods for rolling, otherwise is nan.

        Returns
        ---------
        Feature
            feature instance output
        """
        self.feature = _feature
        self.win = _win
        self.func = _func
        self.kvargs = kvargs
        self.min_periods = _min_periods
        #TODO: why there is no self.data??? check attribute
        # print(self.win, self.func, self.kvargs,self.data)

    def __str__(self):
        s = 'args:'
        for k,v in self.kvargs.items():
            s += '%s=%s:' % (k,v)
        return '%s(%s,%s,%s)' %(type(self).__name__ ,str(self.feature), str(self.win), s)
    
    
    def load_internal(self):
        self.feature.load_feature() # confirm that self.feature's type is Feature
        feature_data = self.feature.data
        # print('check:',feature_data.iloc[-1])
        temp =  getattr(feature_data.groupby(level=0).rolling(self.win, min_periods=self.min_periods), \
                            self.func)(**self.kvargs)
        temp = temp.reset_index(level=0, drop=True)
        return temp

        
class RMax(RollingFeature):
    def __init__(self, _feature, _win):
        super().__init__(_feature, _win, 'max')

class RMin(RollingFeature):
    def __init__(self, _feature, _win):
        super().__init__(_feature, _win, 'min')

class RMedian(RollingFeature):
    def __init__(self, _feature, _win):
        super().__init__(_feature, _win, 'median')

class RMean(RollingFeature):
    def __init__(self, _feature, _win):
        super().__init__(_feature, _win, 'mean')

class RQuant(RollingFeature):
    def __init__(self, _feature, _win):
        super().__init__(_feature, _win, 'quantile')

class RKurt(RollingFeature):
    def __init__(self, _feature, _win):
        super().__init__(_feature, _win, 'kurt')

class RSkew(RollingFeature):
    def __init__(self, _feature, _win):
        super().__init__(_feature, _win, 'skew')

class RSum(RollingFeature):
    def __init__(self, _feature, _win):
        super().__init__(_feature, _win, 'sum')

class RStd(RollingFeature):
    def __init__(self, _feature, _win):
        super().__init__(_feature, _win, 'std')

class RVar(RollingFeature):
    def __init__(self, _feature, _win):
        super().__init__(_feature, _win, 'var')

class Resi(RollingFeature):
    def __init__(self, _feature, _win):
        super().__init__(self, _feature, _win)

    def load_feature(self):
        pass

class EWMA(RollingFeature):
    """EWMA
    by now, just support set span and span=_win
    """
    def __init__(self, _feature, _win):
        """
        parameter
        ---------
        _win means span at the same time.
        """
        super().__init__(_feature, _win, 'ewma')

    def load_internal(self):
        pass

        
#TODO: until now, we have not confirm the index range. so users must confrim different staticFeature have the same index range.
#TODO: until now, not support use previous feature name to define new feature, must use StaticFeature, it is so hard to use.
class DataSet(object):
    def __init__(self, name2expr):
        """tranfrom name2expr to dataset
        params:
        -------
        name2expr should be OrderedDict which can confirm
        output feature's order is the same with input feature's order 
        """
        self.name2expr = name2expr
    
    def eval_expr(self, expr):
        expr = expr.replace('$', 'StaticFeature')
        feature = eval(expr)
        feature.load_feature()
        return feature

    def trans(self):
        feature_dict = OrderedDict()
        for key, val in self.name2expr.items():
            feature_dict[key] = self.eval_expr(val)
        print('eval feature done. start to transform to dataframe..')
        
        data_dict = OrderedDict()
        for k,v in feature_dict.items():
            data_dict[k] = v.data
            print(k,str(v),v.data.index.names)
        df = pd.concat(data_dict, axis=1)
        df = df[list(data_dict.keys())] # let the order of columns the same with input order.
        return df
        

def eval_dataset(name2expr):
    name2expr = OrderedDict(name2expr)
    dataset = DataSet(name2expr)
    df = dataset.trans()
    return df


def unit_test_dataset():
    n2expr = {
        'mv':"$('MonitorValue')",
        'ref_1':"Ref($('MonitorValue'),1)",
        'ref_-1':"Ref($('MonitorValue'),-1)",
        'diff':"Ref($('MonitorValue'),-1) - $('MonitorValue')",
        'max_mv': "RMax($('MonitorValue'),2)",
        'mean_mv':"RMean($('MonitorValue'),2)",
        'mv-mean_mv':"$('MonitorValue') - RMean($('MonitorValue'),2)",
    }
    name2expr = OrderedDict(n2expr)
    dataset = DataSet(name2expr)
    df = dataset.trans()
    print(df.columns)
    print(df.shape)
    print(df)
def unit_test_static_feature():
    sf = StaticFeature('MonitorValue')
    sf.load_feature()
    print(sf.data.name, sf.data.size, str(sf))
    sf.data = None
    sf.load_feature()
    print(sf.data.name, sf.data.size, str(sf))




if __name__ == "__main__":
    # unit_test_static_feature()
    unit_test_dataset()
       
