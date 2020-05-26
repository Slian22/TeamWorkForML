#!/usr/bin/env python 
# -*- coding:utf-8 -*-
from QuickStart_Rhy.api import ipinfo
import pandas as pd
import numpy as np
def getTheNumpy(inputfile):
    example=pd.read_csv(inputfile)
    example=example.values
    return example
def deleteTheCol(cutoff,file,outfile):
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
def deleteTheRow(file,outfile):
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
def FillNaN_PD(inputfile2):#不带标签的Pandas，自行补充
    import numpy as np
    import pandas as pd
    import os
    print(inputfile2)
    # Fill The Data using SimpleImputer
    from sklearn.impute import SimpleImputer
    df = pd.read_csv(inputfile2)
    numpy_Data = np.array(df)
    imp = SimpleImputer(missing_values=np.nan, strategy='mean')
    Sample_data =numpy_Data
    imp.fit(Sample_data)
    pd.DataFrame
    Sample_data_Pandas = pd.DataFrame(Sample_data)
    Sample_data_Pandas.fillna(0, inplace=True)
    return Sample_data_Pandas
def FillNaN_NP(inputfile3):#不带标签的Numpy
    import numpy as np
    import pandas as pd
    import os
    # Fill The Data using SimpleImputer
    from sklearn.impute import SimpleImputer
    df = pd.read_csv(inputfile3)
    numpy_Data = np.array(df)
    imp = SimpleImputer(missing_values=np.nan, strategy='mean')
    #去除第一列数据
    Sample_data =numpy_Data
    imp.fit(Sample_data)
    Sample_data_Pandas = pd.DataFrame(Sample_data)
    Sample_data_Pandas.fillna(0, inplace=True)
    Sample_data_Numpy = np.array(Sample_data_Pandas)
    return Sample_data_Numpy
def CutTheTrain(inputfile,p):#手动切割训练集及测试集
    df=pd.read_csv(inputfile)
    data=df.values
    from random import shuffle
    shuffle(data)
    train = data[:int(len(data) * p), :]
    test = data[int(len(data) * p):, :]
    #return train+test-Numpy
    return train,test
def SMOTE_SAMPLE(inputfile):#返回Numpy
    import pandas as pd
    import numpy as np
    from sklearn import preprocessing
    from sklearn import model_selection
    from sklearn.model_selection import train_test_split
    from imblearn.over_sampling import SMOTE
    churn = FillNaN_PD(inputfile)
    churn_df = pd.read_csv(inputfile)
    feature = list(churn_df.columns.values)  # 提取特征值
    churn.columns = feature
    X_normal = preprocessing.StandardScaler().fit_transform(churn.drop('y', axis=1).to_numpy())
    y = churn['y']
    X_train, X_test, y_train, y_test = train_test_split(X_normal, y, test_size=0.2, random_state=4)#testsize和random_state自己设置
    over_samples = SMOTE(random_state=1234)#random自己设置
    over_samples_X, over_samples_y = over_samples.fit_sample(X_train, y_train.astype('int'))
    # 重抽样前的类别比例
    print("重抽样前的类别比例")
    print(y_train.value_counts() / len(y_train))
    # 重抽样后的类别比例
    print("重抽样后的类别比例")
    print(pd.Series(over_samples_y).value_counts() / len(over_samples_y))
    return over_samples_X,over_samples_y
def LASSO_EXAMPLE(inputfile):
    # 导入使用的模块
    import os
    import numpy as np
    import pandas as pd
    from sklearn.linear_model import Lasso,LassoCV,LassoLarsCV
    from sklearn.linear_model import Lasso
    from keras.models import Sequential  # 有的同学可能会遇到 kernel died，restarting的问题，可参见我的另一片文章
    from keras.layers.core import Dense, Activation
    import matplotlib.pyplot as plt
    # import tensorflow as tf

    #
    data = pd.read_csv(inputfile)
    des = data.describe()
    r = des.T
    r = r[['min', 'max', 'mean', 'std']]
    np.round(r, 2)  # 保留2位小数,四舍六入五留双(五留双即遇五看五前面是偶数则保留，奇数进位)
    np.round(data.corr(method='pearson'), 2)  # method={'pearson','spearman','kendall'},计算相关系数，相关分析
    model = LassoLarsCV(alpha=1)  # LASSO回归的特点是在拟合广义线性模型的同时进行变量筛选和复杂度调整，剔除存在共线性的变量
    model.fit(data.iloc[:, :data.shape[1] - 1], data.iloc[:, data.shape[1] - 1])
    model_coef = pd.DataFrame(pd.DataFrame(model.coef_).T)
    model_coef.columns = ['x%d' % i for i in np.arange(13) + 1]
    print(model_coef)
    # 由系数表可知，系数值为零的要剔除

def PCA_EXAMPLE(inputfile,k):
    from sklearn.decomposition import PCA
    x_std = FillNaN_NP(inputfile)#自定义化数据
    pca=PCA(n_components=k)
    pca.fit(x_std)
    x_trainPCA = pca.transform(x_std)
    print("PCA component shape: {}".format(x_trainPCA.shape))
    print("PCA component components: {}".format(x_trainPCA))
    import matplotlib.pyplot as plt
    plt.matshow(pca.components_,cmap='viridis')
    id_EX=[]
    for i in range(k):
        id_EX.append(str(i)+"component")
    plt.yticks([0,k-1],id_EX)
    plt.colorbar()
    plt.xlabel("Feature")
    plt.ylabel("Principal components")
    return x_trainPCA

