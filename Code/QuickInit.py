#!/usr/bin/env python 
# -*- coding:utf-8 -*-
from QuickStart_Rhy.api import ipinfo
import pandas as pd
import numpy as np
def accuracy_score(truth, pred):
    """ Returns accuracy score for input truth and predictions. """

    # Ensure that the number of predictions matches number of outcomes
    # 确保预测的数量与结果的数量一致
    if len(truth) == len(pred):

        # Calculate and return the accuracy as a percent
        # 计算预测准确率（百分比）
        # 用bool的平均数算百分比
        return (truth == pred).mean() * 100

    else:
        return 0

def fit_model_k_fold(X, y,n):
    """ Performs grid search over the 'max_depth' parameter for a
        decision tree regressor trained on the input data [X, y]. """
    from sklearn.model_selection import train_test_split
    from sklearn.model_selection import GridSearchCV
    from sklearn.model_selection import KFold
    from sklearn.metrics import make_scorer
    from sklearn.tree import DecisionTreeClassifier
    # Create cross-validation sets from the training data
    # cv_sets = ShuffleSplit(n_splits = 10, test_size = 0.20, random_state = 0)
    k_fold = KFold(n_splits=n)

    #  Create a decision tree clf object
    clf = DecisionTreeClassifier(random_state=80)

    params = {'max_depth': range(1, 21), 'criterion': np.array(['entropy', 'gini'])}

    # Transform 'accuracy_score' into a scoring function using 'make_scorer'
    scoring_fnc = make_scorer(accuracy_score)

    # Create the grid search object
    grid = GridSearchCV(clf, param_grid=params, scoring=scoring_fnc, cv=k_fold)
    result=pd.DataFrame(grid.cv_results_)
    # Fit the grid search object to the data to compute the optimal model
    grid = grid.fit(X, y)
    print(result)
    print("Best estimator:\n{}".format(grid.best_estimator_))
    print("Best Score:\n{}".format(grid.best_score_))
    best_model = grid.best_estimator_
    print('测试集上准确率：', best_model.score(X, y))
    print()
    # Return the optimal model after fitting the data
    return grid.best_estimator_
def predict_4(X, Y):
    from sklearn import tree
    clf = tree.DecisionTreeClassifier()#设置参数参照readme
    clf = clf.fit(X, Y)
    return clf
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
    over_samples_X, over_samples_y = over_samples.fit_sample(X_train, y_train.astype('float'))
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
def Corr(inputfile):#尽量传入的数据相对维数较少
    import matplotlib.pyplot as plt
    import seaborn as sns
    tips = pd.read_csv(inputfile)
    print("相关性：")
    print(tips.corr())
    # 相关性热力图
    print("相关性热力图：")
    print(sns.heatmap(tips.corr()))
    # 分层相关性热力图
    print("分层相关性热力图：")
    print(sns.clustermap(tips.corr()))
    return True
def MakeTree_1(inputfile):
    import matplotlib.pyplot as plt
    from sklearn.datasets import load_iris
    from sklearn.datasets import load_breast_cancer
    from sklearn import metrics
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import train_test_split
    from sklearn import tree
    from sklearn.preprocessing import MinMaxScaler
    from sklearn import preprocessing
    df = FillNaN_PD(inputfile)
    pd_Example = pd.read_csv(inputfile)
    feature = list(pd_Example.columns.values)  # 提取特征值
    feature2=list(pd_Example.columns.values)# 提取特征值
    del feature2[-1]# 提取特征值-1用于graphviz
    df.columns = feature
    target = df['y']
    X_normal = preprocessing.StandardScaler().fit_transform(df.drop('y', axis=1).to_numpy())
    X_train, X_test, y_train, y_test = train_test_split(X_normal, target, test_size=0.2, random_state=4)
    ### 调参部分
    clf = DecisionTreeClassifier()
    '''
    参照readme.md
    criterion = "gini",
    splitter = "best",
    max_depth = None,
    min_samples_split = 2,
    min_samples_leaf = 1,
    min_weight_fraction_leaf = 0.,
    max_features = None,
    random_state = None,
    max_leaf_nodes = None,
    min_impurity_decrease = 0.,
    min_impurity_split = None,
    class_weight = None,
    presort = 'deprecated',
    ccp_alpha = 0.0):
    '''
    ###
    clf.fit(X_train, y_train.astype('float'))
    from sklearn.tree import export_graphviz
    export_graphviz(clf,out_file="tree.dot",feature_names=feature2,impurity=False,filled=True)
    import graphviz
    with open("tree.dot") as f:
        dot_graph=f.read()
    display(graphviz.Source(dot_graph))
    print(plt.show(tree.plot_tree(clf)))
    pred = clf.predict(X_test)
    print(metrics.classification_report(y_test, pred))
    print(metrics.accuracy_score(y_test, pred))
    y_score = clf.predict_proba(X_test)[:, 1]
    fpr, tpr, threshold = metrics.roc_curve(y_test, y_score)
    roc_auc = metrics.auc(fpr, tpr)
    # 绘制面积图
    plt1 = plt.stackplot(fpr, tpr, color='steelblue', alpha=0.5, edgecolor='black')
    # 添加边际线
    plt.plot(fpr, tpr, color='black', lw=1)
    # 添加对角线
    plt.plot([0, 1], [0, 1], color='red', linestyle='--')
    # 添加文本信息
    plt.text(0.5, 0.3, 'ROC curve (area = %0.3f)' % roc_auc)
    # 添加x轴与y轴标签
    plt.xlabel('1-Specificity')
    plt.ylabel('Sensitivity')
    # 显示图形
    print(plt.show())
    from sklearn.metrics import precision_recall_curve
    precision, recall, thresholds = precision_recall_curve(y_test, clf.predict(X_test))
    from sklearn.datasets import make_blobs
    # find threshold closest to zero
    close_zero = np.argmin(np.abs(thresholds))
    plt.plot(precision[close_zero], recall[close_zero], 'o', markersize=10,
             label="threshold zero", fillstyle="none", c='k', mew=2)
    plt.plot(precision, recall, label="precision recall curve")
    plt.xlabel("Precision")
    plt.ylabel("Recall")
    plt.legend(loc="best")
    print(plt.show())
    from sklearn.metrics import roc_curve
    from sklearn.linear_model import LogisticRegression
    model = LogisticRegression(solver='newton-cg', multi_class='ovr')
    model.fit(X_train, y_train)
    y_pre = model.predict_proba(X_test)
    y_0 = list(y_pre[:, 1])
    fpr, tpr, thresholds = roc_curve(y_test, y_0)  # 计算fpr,tpr,thresholds
    # 计算ks
    KS_max = 0
    best_thr = 0
    for i in range(len(fpr)):
        if (i == 0):
            KS_max = tpr[i] - fpr[i]
            best_thr = thresholds[i]
        elif (tpr[i] - fpr[i] > KS_max):
            KS_max = tpr[i] - fpr[i]
            best_thr = thresholds[i]

    print('最大KS为：', KS_max)
    print('最佳阈值为：', best_thr)
    fpr, tpr, thresholds = roc_curve(y_test, y_0)
    return True
def Cross_Validation(inputfile,n):
    from sklearn import tree
    from sklearn.model_selection import train_test_split
    df = FillNaN_PD(inputfile)
    pd_Example = pd.read_csv(inputfile)
    feature = list(pd_Example.columns.values)
    df.columns = feature #添加卡特征值
    out = df['y']
    features = df.drop('y', axis=1)
    X_train, X_test, y_train, y_test = train_test_split(features, out, test_size=0.2, random_state=0)##自行定义即可
    clf = fit_model_k_fold(X_train, y_train,n)
    from IPython.display import Image
    import pydotplus
    dot_data = tree.export_graphviz(clf, out_file=None,
                                    class_names=['0', '1'],
                                    filled=True, rounded=True,
                                    special_characters=True)
    graph = pydotplus.graph_from_dot_data(dot_data)
    display(Image(graph.create_png()))
    display(graph)
    print(graph)
    print ("k_fold Parameter 'max_depth' is {} for the optimal model.".format(clf.get_params()['max_depth']))
    print("k_fold Parameter 'criterion' is {} for the optimal model.".format(clf.get_params()['criterion']))

    return True
def Simplecross_val_score(inputfile):
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.model_selection import cross_val_score
    from sklearn.model_selection import train_test_split
    df = FillNaN_PD(inputfile)
    pd_Example = pd.read_csv(inputfile)
    feature = list(pd_Example.columns.values)
    df.columns = feature  # 添加卡特征值
    out = df['y']
    features = df.drop('y', axis=1)
    X_train, X_test, y_train, y_test = train_test_split(features, out, test_size=0.2, random_state=0)  ##自行定义即可
    k_range = [2, 4, 5, 10]  # k选择2，4，5，10四个参数
    cv_scores = []  # 分别放用4个参数训练得到的精确度
    for k in k_range:
        knn = KNeighborsClassifier(n_neighbors=k)
        # *****下面这句进行了交叉验证**********
        scores = cross_val_score(knn, X_train, y_train, cv=3)  # 进行3折交叉验证，返回的是3个值即每次验证的精确度
        cv_score = np.mean(scores)  # 把某个k对应的精确度求平均值
        print('k={}，验证集上的准确率={:.3f}'.format(k, cv_score))
        cv_scores.append(cv_score)
    import matplotlib.pyplot as plt
    from sklearn.model_selection import validation_curve
    from sklearn.svm import SVC
    c_range = [1e-3, 1e-2, 0.1, 1, 10, 100, 1000, 10000]  # C的取值范围

    train_scores, test_scores = validation_curve(SVC(kernel='linear'), X_train, y_train, param_name='C',
                                                 param_range=c_range, cv=5,
                                                 scoring='accuracy')  # 通过验证曲线得到不同取值的C在验证集合训练集上的得分。
    train_scores_mean = np.mean(train_scores,
                                axis=1)  # 对每一个参数C，他通过5折交叉验证后都会得到5个得分，这里就是把这5个得分求均值。求得的均值用于接下来的绘制每个参数的取值对应的得分。
    train_scores_std = np.std(train_scores, axis=1)  # 求标准差是为了画出它的置信区间
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    plt.figure(figsize=(10, 8))
    plt.title('Validation Curve with SVM')
    plt.xlabel('C')
    plt.ylabel('Score')
    plt.ylim(0.0, 1.1)
    lw = 2
    plt.semilogx(c_range, train_scores_mean, label="Training score",
                 color="darkorange", lw=lw)
    plt.fill_between(c_range, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.2,
                     color="darkorange", lw=lw)  # 画出置信区间
    plt.semilogx(c_range, test_scores_mean, label="Cross-validation score",
                 color="navy", lw=lw)
    plt.fill_between(c_range, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.2,
                     color="navy", lw=lw)
    plt.legend(loc="best")
    plt.show()
    return True
def Sample_RandomSearch(inputfile):
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.model_selection import cross_val_score
    import numpy as np
    import pandas as pd
    import xgboost as xgb
    from sklearn.model_selection import RandomizedSearchCV
    from sklearn.model_selection import train_test_split
    df = FillNaN_PD(inputfile)
    pd_Example = pd.read_csv(inputfile)
    feature = list(pd_Example.columns.values)
    df.columns = feature  # 添加卡特征值
    out = df['y']
    trainlabel=df['y']
    features = df.drop('y', axis=1)
    X_train, X_test, y_train, y_test = train_test_split(features, out, test_size=0.2, random_state=0)  ##自行定义即可
    # 分类器使用 xgboost
    clf1 = xgb.XGBClassifier()

    # 设定搜索的xgboost参数搜索范围，值搜索XGBoost的主要6个参数
    param_dist = {
        'criterion': np.array(['entropy', 'gini']),
        'n_estimators': range(80, 200, 4),
        'max_depth': range(2, 15, 1),
        'learning_rate': np.linspace(0.01, 2, 20),
        'subsample': np.linspace(0.7, 0.9, 20),
        'colsample_bytree': np.linspace(0.5, 0.98, 10),
        'min_child_weight': range(1, 9, 1)
    }
    # RandomizedSearchCV参数说明，clf1设置训练的学习器
    # param_dist字典类型，放入参数搜索范围
    # scoring = 'neg_log_loss'，精度评价方式设定为“neg_log_loss“
    # n_iter=300，训练300次，数值越大，获得的参数精度越大，但是搜索时间越长
    # n_jobs = -1，使用所有的CPU进行训练，默认为1，使用1个CPU
    grid = RandomizedSearchCV(clf1, param_dist, cv=3, scoring='neg_log_loss', n_iter=300, n_jobs=-1)
    # 在训练集上训练
    grid.fit(X_train.values, np.ravel(y_train.values))
    # 返回最优的训练器
    best_estimator = grid.best_estimator_
    print(best_estimator)
    # 输出最优训练器的精度
    print(grid.best_score_)
    return True