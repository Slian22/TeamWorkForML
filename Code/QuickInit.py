#!/usr/bin/env python 
# -*- coding:utf-8 -*-
import pandas as pd
import numpy as np
from IPython.display import *
def plot_ks(y_test, y_score, positive_flag):#绘制索引
    import matplotlib.pyplot as plt
    # 对y_test,y_score重新设置索引
    y_test.index = np.arange(len(y_test))
    # y_score.index = np.arange(len(y_score))
     # 构建目标数据集
    target_data = pd.DataFrame({'y_test': y_test, 'y_score': y_score})
     # 按y_score降序排列
    target_data.sort_values(by='y_score', ascending=False, inplace=True)
    # 自定义分位点
    cuts = np.arange(0.1, 1, 0.1)
    # 计算各分位点对应的Score值
    index = len(target_data.y_score) * cuts
    scores = target_data.y_score.iloc[index.astype('int')]
    # 根据不同的Score值，计算Sensitivity和Specificity
    Sensitivity = []
    Specificity = []
    for score in scores:
        # 正例覆盖样本数量与实际正例样本量
        positive_recall = \
        target_data.loc[(target_data.y_test == positive_flag) & (target_data.y_score > score), :].shape[0]
        positive = sum(target_data.y_test == positive_flag)
        # 负例覆盖样本数量与实际负例样本量
        negative_recall = \
            target_data.loc[(target_data.y_test != positive_flag) & (target_data.y_score <= score), :].shape[0]
        negative = sum(target_data.y_test != positive_flag)
        Sensitivity.append(positive_recall / positive)
        Specificity.append(negative_recall / negative)
        # 构建绘图数据
    plot_data = pd.DataFrame({'cuts': cuts, 'y1': 1 - np.array(Specificity), 'y2': np.array(Sensitivity),
                              'ks': np.array(Sensitivity) - (1 - np.array(Specificity))})
    # 寻找Sensitivity和1-Specificity之差的最大值索引
    max_ks_index = np.argmax(plot_data.ks)
    plt.plot([0] + cuts.tolist() + [1], [0] + plot_data.y1.tolist() + [1], label='1-Specificity')
    plt.plot([0] + cuts.tolist() + [1], [0] + plot_data.y2.tolist() + [1], label='Sensitivity')
    # 添加参考线
    plt.vlines(plot_data.cuts[max_ks_index], ymin=plot_data.y1[max_ks_index],
               ymax = plot_data.y2[max_ks_index], linestyles = '--')
    # 添加文本信息
    plt.text(x=plot_data.cuts[max_ks_index] + 0.01,
             y = plot_data.y1[max_ks_index] + plot_data.ks[max_ks_index] / 2,
             s = 'KS= %.2f' % plot_data.ks[max_ks_index])
    # 显示图例
    plt.legend()
    # 显示图形
    plt.show()
    return True
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
    # Fit the grid search object to the data to compute the optimal model
    grid = grid.fit(X, y)
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
    #运行这个函数即为自定义删除缺失为（1-cutoff）*100%的列
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
    import os
    df = pd.read_csv(file)
    print("一共有：",df.shape[1]-1,"列")
    rows_not_null = df.count(axis=1)
    df['NotNumber'] = rows_not_null/df.shape[1]
    if(os.path.exists(outfile)):
        os.remove(outfile)
    df.to_csv(outfile)
    return True
def FillNaN_PD(inputfile2):#不带标签的Pandas，自行补充
    import numpy as np
    import pandas as pd
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
def SMOTE_SAMPLE(inputfile,n):#返回Numpy
    import pandas as pd
    from sklearn.model_selection import train_test_split
    from imblearn.over_sampling import SMOTE
    n_1=float(1/n)
    churn = FillNaN_PD(inputfile)
    churn_df = pd.read_csv(inputfile)
    feature = list(churn_df.columns.values)  # 提取特征值
    churn.columns = feature
    X_normal = churn.drop('y', axis=1).to_numpy()
    y = churn['y']
    X_train, X_test, y_train, y_test = train_test_split(X_normal, y, test_size=0.25, random_state=4)#testsize和random_state自己设置
    over_samples = SMOTE(sampling_strategy=n_1,random_state=123)#random自己设置
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
    from sklearn import preprocessing
    x_std = FillNaN_PD(inputfile)#自定义化数据
    x_std = preprocessing.StandardScaler().fit_transform(x_std.to_numpy())
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
def FindPCA_N(inputfile):
    from sklearn.decomposition import PCA
    from sklearn import preprocessing
    from matplotlib import pyplot as plt
    x_std = FillNaN_PD(inputfile)  # 自定义化数据
    x_std = preprocessing.StandardScaler().fit_transform(x_std.to_numpy())
    pca = PCA()
    pca.fit(x_std)
    ratio = pca.explained_variance_ratio_
    print("pca.components_", pca.components_.shape)
    print("pca_var_ratio", pca.explained_variance_ratio_.shape)
    # 绘制图形
    plt.plot([i for i in range(x_std.shape[1])],
             [np.sum(ratio[:i + 1]) for i in range(x_std.shape[1])])
    plt.xticks(np.arange(x_std.shape[1], step=5))
    plt.yticks(np.arange(0, 1.01, 0.05))
    plt.ylabel('cumulative explained variance')
    plt.xlabel('number of components after PCA dimension reduction')
    plt.grid()
    plt.show()
    return True
def Corr(inputfile):#尽量传入的数据相对维数较少
    import seaborn as sns
    tips =FillNaN_PD(inputfile)
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
    from sklearn import metrics
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.model_selection import train_test_split
    from sklearn import tree
    df = FillNaN_PD(inputfile)
    pd_Example = pd.read_csv(inputfile)
    feature = list(pd_Example.columns.values)  # 提取特征值
    feature2=list(pd_Example.columns.values)# 提取特征值
    del feature2[-1]# 提取特征值-1用于graphviz
    df.columns = feature
    target = df['y']
    X_normal = df.drop('y', axis=1).to_numpy()
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
    import os
    if os.path.exists('tree.dot'):
        os.remove('tree.dot')
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
    precision, recall, thresholds = precision_recall_curve(y_test, clf.predict_proba(X_test)[:,1])
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
    y_pre = clf.predict_proba(X_test)
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
    #绘制KS曲线
    # 自定义绘制ks曲线的函数
    plot_ks(y_test,y_score,1)
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
def Simplecross_val_score(inputfile,n):
    from sklearn.model_selection import cross_val_score
    from sklearn.model_selection import train_test_split
    from sklearn.tree import DecisionTreeClassifier
    df = FillNaN_PD(inputfile)
    pd_Example = pd.read_csv(inputfile)
    feature = list(pd_Example.columns.values)
    df.columns = feature  # 添加卡特征值
    out = df['y']
    features = df.drop('y', axis=1)
    X_train, X_test, y_train, y_test = train_test_split(features, out, test_size=0.25, random_state=0)  ##自行定义即可
    knn = DecisionTreeClassifier()
    knn.fit(X_train, y_train.astype('float'))#构建决策树
    # *****下面这句进行了交叉验证**********
    scores = cross_val_score(knn, features, out, cv=n)  # 进行n折交叉验证，返回的是n个值即每次验证的精确度
    cv_score = np.mean(scores)  # 把某个k对应的精确度求平均值
    print('平均的准确率={:.3f}'.format(cv_score))
    print('std={:.3f}'.format(np.std(scores)))
    return True
def Sample_RandomSearch(inputfile):
    from sklearn.tree import DecisionTreeClassifier
    import numpy as np
    import pandas as pd
    from sklearn.model_selection import RandomizedSearchCV
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import classification_report
    df = FillNaN_PD(inputfile)
    pd_Example = pd.read_csv(inputfile)
    feature = list(pd_Example.columns.values)
    df.columns = feature  # 添加卡特征值
    out = df['y']
    trainlabel=df['y']
    features = df.drop('y', axis=1)
    X_train, X_test, y_train, y_test = train_test_split(features, out, test_size=0.25, random_state=0)  ##自行定义即可
    tuned_parameters = {'max_depth': range(1, 21), 'criterion': np.array(['entropy', 'gini'])}#指数分布
    clf = RandomizedSearchCV(DecisionTreeClassifier(), tuned_parameters, cv=10,
                             scoring="accuracy", n_iter=100)
    clf.fit(X_train, y_train)
    print("Best parameters set found:", clf.best_params_)
    print("Randomized Grid scores:")
    #     for params, mean_score, scores in clf.fit_params,clf.mean_score,clf.score:
    #         print("\t%0.3f (+/-%0.03f) for %s" % (mean_score, scores() * 2, params))
    #     print("\t%0.3f (+/-%0.03f) for %s" % (clf.mean_score,clf.score * 2, clf.fit_params))
    print(clf)
    print("Optimized Score:", clf.score(X_test, y_test))
    print("Detailed classification report:")
    y_true, y_pred = y_test, clf.predict(X_test)
    print(classification_report(y_true, y_pred))
    return True
def MakeTree_2(inputfile):
    import matplotlib.pyplot as plt
    from sklearn import metrics
    from sklearn.ensemble  import RandomForestClassifier
    from sklearn.model_selection import train_test_split
    df = FillNaN_PD(inputfile)
    pd_Example = pd.read_csv(inputfile)
    feature = list(pd_Example.columns.values)  # 提取特征值
    feature2=list(pd_Example.columns.values)# 提取特征值
    del feature2[-1]# 提取特征值-1用于graphviz
    df.columns = feature
    target = df['y']
    X_normal = df.drop('y', axis=1).to_numpy()
    X_train, X_test, y_train, y_test = train_test_split(X_normal, target, test_size=0.2, random_state=4)
    ### 调参部分
    clf = RandomForestClassifier()
    '''
    参照readme.md
    默认值
                n_estimators=100,
                 criterion="gini",
                 max_depth=None,
                 min_samples_split=2,
                 min_samples_leaf=1,
                 min_weight_fraction_leaf=0.,
                 max_features="auto",
                 max_leaf_nodes=None,
                 min_impurity_decrease=0.,
                 min_impurity_split=None,
                 bootstrap=True,
                 oob_score=False,
                 n_jobs=None,
                 random_state=None,
                 verbose=0,
                 warm_start=False,
                 class_weight=None,
                 ccp_alpha=0.0,
                 max_samples=None
    '''
    ###
    s=clf.fit(X_train, y_train.astype('float'))
    pred = clf.predict(X_test)
    print(s)
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
    precision, recall, thresholds = precision_recall_curve(y_test, clf.predict_proba(X_test)[:,1])
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
    y_pre = clf.predict_proba(X_test)
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
    #绘制KS曲线
    # 自定义绘制ks曲线的函数
    plot_ks(y_test,y_score,1)
    return True
def SIMPLE_XGB(inputfile):
    from matplotlib import pyplot
    from sklearn.model_selection import train_test_split
    from xgboost import XGBClassifier
    from sklearn.metrics import accuracy_score
    df = FillNaN_PD(inputfile)
    pd_Example = pd.read_csv(inputfile)
    feature = list(pd_Example.columns.values)  # 提取特征值
    feature2 = list(pd_Example.columns.values)  # 提取特征值
    del feature2[-1]  # 提取特征值-1用于graphviz
    df.columns = feature
    target = df['y']
    X_normal = df.drop('y', axis=1).to_numpy()
    X_train, X_test, y_train, y_test = train_test_split(X_normal, target, test_size=0.2, random_state=4)
    num_round = 300
    bst1 = XGBClassifier(max_depth=5, learning_rate=1, n_estimators=num_round,  # 弱分类树太少的话取不到更多的特征重要性
                         silent=True, objective='binary:logistic')
    bst1.fit(X_train, y_train)
    from sklearn.model_selection import StratifiedKFold
    from sklearn.model_selection import cross_val_score
    preds = bst1.predict(X_train)
    print(preds)
    test_accuracy = accuracy_score(y_train, preds)
    print("Test Accuracy: %.2f%%" % (test_accuracy * 100.0))
    from xgboost import plot_importance  # 显示特征重要性
    plot_importance(bst1)  # 打印重要程度结果。
    pyplot.show()
    num_round = 100
    bst2 = XGBClassifier(max_depth=2, learning_rate=0.1, n_estimators=num_round, silent=True,
                         objective='binary:logistic')
    bst2.fit(X_train, y_train)
    kfold = StratifiedKFold(n_splits=10, random_state=7)
    results = cross_val_score(bst2, X_train, y_train, cv=kfold)  # 对数据进行十折交叉验证--9份训练，一份测试
    print(results)
    print("CV Accuracy: %.2f%% (%.2f%%)" % (results.mean() * 100, results.std() * 100))
    from xgboost import plot_importance  # 显示特征重要性
    plot_importance(bst2)  # 打印重要程度结果。
    pyplot.show()
    return True