#!/usr/bin/env python 
# -*- coding:utf-8 -*-
from QuickStart_Rhy.api import ipinfo
import pandas as pd
def deleteTheCol(cutoff=0.5,file="sample/model_sample.csv",outfile="sample/Pre_Sample.csv"):
    #运行这个函数即为自定义删除缺失为（cutoff*100）%的列
    import pandas as pd
    import os
    inputfile=file
    df=pd.read_csv(inputfile)
    for i in  [column for column in df]:
        n = len(df)
        cnt = df[i].count()
        if (float(cnt) / n) < cutoff:
            df.drop(i, axis=1, inplace=True)
    if (os.path.exists(outfile)):
        os.remove(outfile)
    df.to_csv(outfile)
    return True
def Sample_NotNull(file="sample/model_sample.csv",outfile="sample/model_sample_NotNumber.csv"):
    #自行删除空值多的行。
    import pandas as pd
    import numpy as np
    import os
    df = pd.read_csv(file)
    print("一共有：",df.shape[0]-1,"列")
    rows_not_null = df.count(axis=1)
    df['NotNumber'] = rows_not_null/df.shape[1]
    if(os.path.exists(outfile)):
        os.remove(outfile)
    df.to_csv(outfile)
    return True
def FillNaN_PD(inputfile2):
    import numpy as np
    import pandas as pd
    import os
    print(inputfile2)
    # Fill The Data using SimpleImputer
    from sklearn.impute import SimpleImputer
    df = pd.read_csv(inputfile2)
    numpy_Data = np.array(df)
    imp = SimpleImputer(missing_values=np.nan, strategy='mean')
    Sample_data = np.delete(numpy_Data, 0, axis=1)
    imp.fit(Sample_data)
    Sample_data_Pandas = pd.DataFrame(Sample_data)
    Sample_data_Pandas.fillna(0, inplace=True)
    return Sample_data_Pandas
def FillNaN_NP(file='sample/model_sample.csv'):
    import numpy as np
    import pandas as pd
    import os
    # Fill The Data using SimpleImputer
    from sklearn.impute import SimpleImputer
    df = pd.read_csv(file)
    numpy_Data = np.array(df)
    imp = SimpleImputer(missing_values=np.nan, strategy='mean')
    #去除第一列数据
    Sample_data = np.delete(numpy_Data, 0, axis=1)
    imp.fit(Sample_data)
    Sample_data_Pandas = pd.DataFrame(Sample_data)
    Sample_data_Pandas.fillna(0, inplace=True)
    Sample_data_Numpy = np.array(Sample_data_Pandas)
    return Sample_data_Numpy
def CutTheTrain(inputfile='sample/model_sample.csv',p=0.8):
    df=pd.read_csv(inputfile)
    data=df.values
    from random import shuffle
    shuffle(data)
    train = data[:int(len(data) * p), :]
    test = data[int(len(data) * p):, :]
    #return train+test-Numpy
    return train,test
def SMOTE(inputfile='sample/model_sample.csv'):
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    from sklearn import model_selection
    from sklearn import tree
    from sklearn import metrics
    from imblearn.over_sampling import SMOTE
    data=pd.read_csv(inputfile)
    return True
#def MakeTree(criterion='gini',)