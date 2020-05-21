#!/usr/bin/env python 
# -*- coding:utf-8 -*-

def FillNaN_PD():
    import numpy as np
    import pandas as pd
    import os
    # Fill The Data using SimpleImputer
    from sklearn.impute import SimpleImputer

    df = pd.read_csv('sample/model_sample.csv', header=0)
    numpy_Data = np.array(df)
    imp = SimpleImputer(missing_values=np.nan, strategy='mean')
    Sample_data = np.delete(numpy_Data, 0, axis=1)
    imp.fit(Sample_data)
    Sample_data_Pandas = pd.DataFrame(Sample_data)
    Sample_data_Pandas.fillna(0, inplace=True)
    return Sample_data_Pandas
def FillNaN_NP():
    import numpy as np
    import pandas as pd
    import os
    # Fill The Data using SimpleImputer
    from sklearn.impute import SimpleImputer
    df = pd.read_csv('sample/model_sample.csv', header=0)
    numpy_Data = np.array(df)
    imp = SimpleImputer(missing_values=np.nan, strategy='mean')
    Sample_data = np.delete(numpy_Data, 0, axis=1)
    imp.fit(Sample_data)
    Sample_data_Pandas = pd.DataFrame(Sample_data)
    Sample_data_Pandas.fillna(0, inplace=True)
    Sample_data_Numpy = np.array(Sample_data_Pandas)
    return Sample_data_Numpy