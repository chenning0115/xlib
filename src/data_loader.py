# @author charnix@pku.edu.cn

import os
import sys
import numpy as np
import pandas as pd
from datetime import datetime


class XDataLoader(object):
    """DataLoader:
        Tools for Loading Data from Disk 
    """

    def __init__(self):
        
        # dataset
        self.data = self.load_data()
        self.columns = set(self.data.columns)

    def load_data(self):
        """load_data
         real load function, need to implement
         Be Careful: the loaded data must confim that Timestamp is uniform, In other words, no missing timestamp.
         Returns
         --------
         data: pandas DataFrame Instance
            which index must be multi-index(id, DataTime)

        """
        pass

    def load_data_static_feature(self, _field_name):
        """
        """
        if _field_name in self.columns:
            return self.data[_field_name]
        else:
            raise Exception('%s does not exists in raw data which columns is %s'\
                 % (_field_name, str(self.columns)))
    
class MonitorValueXDataLoader(XDataLoader):
    """MonitorValueXDataLoader
    used to load monitor data of gas of coal mine.
    """

    def __init__(self, _path_prefix):
        self.path_prefix = _path_prefix

        self.data_file = '044A02上隅角.max.pkl'
        self.start_str = '2017-09-15 00:00:00'
        self.end_str = '2018-05-01 00:00:00'
        self.freq = '1min'

        self.path_data = os.path.join(self.path_prefix, self.data_file)
        super().__init__()
    
    def format_index(self, data):
        """format_index
        format data level-2 index(datetime) 

        Parameters:
        -----------
        data pd.DataFrame
            has multi-index(SensorID, Time)
        """
        id_list = set(data.index.get_level_values(level=0))
        tr = pd.date_range(start=pd.Timestamp(self.start_str), end=pd.Timestamp(self.end_str), freq=self.freq)
        dd = {ai:pd.DataFrame(index=tr) for ai in id_list}
        df = pd.concat(dd, axis=0)
        df.index.names = list(data.index.names)
        df = pd.merge(left=df, right=data, left_index=True, right_index=True, how='left')
        return df
        

    def load_data(self):
        data = pd.read_pickle(self.path_data)
        data = data.reset_index()
        data['Time'] = data['Time'].apply(lambda x: pd.Timestamp(x))
        data = data.sort_values(['SensorID', 'Time']).set_index(['SensorID', 'Time'], drop=True)
        data = self.format_index(data)
        print('Load rawdata from disk done. path=%s' % self.path_data)
        return data


from config import path_data_prefix
xdata_loader = MonitorValueXDataLoader(path_data_prefix)

def unit_test():
    print(xdata_loader.load_data_static_feature('MonitorValue'))



if __name__ == "__main__":
    unit_test()