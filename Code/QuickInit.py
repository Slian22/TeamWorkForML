#!/usr/bin/env python 
# -*- coding:utf-8 -*-
def deleteTheCol(cutoff=0.5,file='sample/model_sample.csv',outfile='sample/Pre_Sample.csv'):
    #运行这个函数即为自定义删除缺失为（cutoff*100）%的列
    import pandas as pd
    import os
    inputfile=file
    df=pd.read_csv(inputfile)
    for i in  [column for column in df]:
        n = len(df)
        cnt = df[i].count()
        if (float(cnt) / n) < cutoff:
            df.drop(i, axis=1, inplace=1)
    os.path.exists(outfile)
    os.remove(outfile)
    df.to_csv(outfile)
def deleteTheRow(cutoff=0.5,file='sample/model_sample.csv',outfile='sample/Pre2_Sample.csv'):
    import pandas as pd
    import os
    inputfile = file
    df = pd.read_csv(inputfile)
    Row = len(df) - 1  # 列数
    id=[]
    for i in [index for index in df]:
        id.append(i)
        print(i)
    rows_not_null = df.count(axis=1)
    pd1 = pd.DataFrame(rows_not_null)
    rows_NotNull = pd.DataFrame(rows_not_null)
    for i in (1,len(id)):
        notnullnum=pd1.iloc[0, i-1]-1
        if ((Row-notnullnum)/Row) >cutoff :
            df.drop[i-1]
    os.path.exists(outfile)
    os.remove(outfile)
    df.to_csv(outfile)

def FillNaN_PD():
    import numpy as np
    import pandas as pd
    import os
    # Fill The Data using SimpleImputer
    from sklearn.impute import SimpleImputer
    df = pd.read_csv('sample/Pre_Sample.csv')
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
    #去除第一列数据
    Sample_data = np.delete(numpy_Data, 0, axis=1)
    imp.fit(Sample_data)
    Sample_data_Pandas = pd.DataFrame(Sample_data)
    Sample_data_Pandas.fillna(0, inplace=True)
    Sample_data_Numpy = np.array(Sample_data_Pandas)
    return Sample_data_Numpy