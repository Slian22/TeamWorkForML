

Machine Learning With Python For TeamWork

![1](img/)

1.jpg需要最后做完绘制



![2](img/2.jpg)

# 任务清单

## 1、数据预处理方法

### 1.1 删除列缺失值较多的数据（自定义`CutOff`）

```python
  def deleteTheCol(cutoff,inputfile,outputfile):
      return True
  #c传入输入文件，输出文件路径；cutoff为自定义删除缺失为(1-(cutoff*100))%的列
```

- [x] 已实现

### 1.2 删除行缺失值较多的数据（程序生成`CSV`，手动格式删除）[可选]

```python
  def deleteTheRow(file,outfile):
      return True
  #传入输入文件，输出文件路径；会生成比原先数据多一列的数据，值越大表明缺失的数据越少
```

- [x] 已实现

### 1.3 缺失值填充（`SimpleImputer`{或者插值法或者均值/中位数/众数插补}+填充0值）

本项目采取`mean`值填充，若想修改方法去`QuickInit`中的`FillNaN_PD`修改

```python
  def FillNaN_PD(inputfile):#不带标签的Pandas，自行补充
      return DataFrame
  #传入输入文件，返回不带标签的DataFrame类型数据
  '''
  若想添加标签
  pd_Example=pd.read_csv(inputfile)
  feature = list(pd_Example.columns.values)  # 提取特征值
  pd_Example.columns = feature
  '''
```

 ```python
  def FillNaN_NP(inputfile3):#不带标签的Numpy,自行补充
      return series
  #Same as FillNaN_PD
 ```

- [x] 已实现

### 1.4 异常值处理（删除/平均值修正)

- [ ] 异常值处理

  - 简单统计

  - 3∂原则

  - 箱型图

  - 基于模型检验

  - 基于近邻度的离群点检测

  - .....

    检测到了异常值，我们需要对其进行一定的处理。而一般异常值的处理方法可大致分为以下几种：

    - **删除含有异常值的记录**：直接将含有异常值的记录删除；
    - **视为缺失值**：将异常值视为缺失值，利用缺失值处理的方法进行处理；
    - **平均值修正**：可用前后两个观测值的平均值修正该异常值；
    - **不处理**：直接在具有异常值的数据集上进行数据挖掘；

### 1.4 one hot encode编码定义

- [x] 没写在`QuickInit`里；写在了`HandleTheQuestion`里

  ```python
  from sklearn.preprocessing import OneHotEncoder
  ....
  ```

### 1.5 降维、特征提取

- [x] 主成分分析 `PCA`

  ```python
  def PCA_EXAMPLE(inputfile,k):#返回Numpy，k等于你想降到多少维度
      return x_trainPCA
  # 会输出系数
  #包括主成分的热图
  ```

  ```python
  def Corr(inputfile):#尽量传入的数据相对维数较少
      return True
  # 相关性+相关性热力图+分层相关性热力图
  # 依赖['seaborn','matplotlib']
  ```

### *1.6 `LASSO`

```python
  from sklearn.linear_model import Lasso,LassoCV,LassoLarsCV
  data.corr()
  ....
```
- [ ] 未应用，已实现

## 2、解决非平衡数据问题

### 2.1 `SMOTE`算法

- [x] `SMOTE`，也可自己手动划分数据集

  ```python
  def CutTheTrain(inputfile,p):#手动切割训练集及测试集、p=训练集的比例：0~1
       return train,test
  '''
  返回两个numpy
  train，test
  '''
  ```

```python
def SMOTE_SAMPLE(inputfile):#返回Numpy
    return over_samples_X,over_samples_y
'''
testsize和random_state自己设置
random自己设置
'''
```

## 3、决策树

### 调参`gini`或者`entropy` 树深节点数 

| 参数                                               | `DecisionTreeClassifier`                                     |
| :------------------------------------------------- | :----------------------------------------------------------- |
| 特征选择标准`criterion`                            | 可以使用"`gini`"或者"`entropy`"，前者代表基尼系数，后者代表信息增益。一般说使用默认的基尼系数"`gini`"就可以了，即CART算法。除非你更喜欢类似`ID3`, `C4.5`的最优特征选择方法。 |
| 特征划分点选择标准`splitter`                       | 可以使用"`best`"或者"`random`"。前者在特征的所有划分点中找出最优的划分点。后者是随机的在部分划分点中找局部最优的划分点。默认的"`best`"适合样本量不大的时候，而如果样本数据量非常大，此时决策树构建推荐"`random`" |
| 划分时考虑的最大特征数`max_features`               | 可以使用很多种类型的值，默认是"None",意味着划分时考虑所有的特征数；如果是"`log2`"意味着划分时最多考虑`log2N`个特征；如果是"`sqrt`"或者"`auto`"意味着划分时最多考虑N−−√N个特征。如果是整数，代表考虑的特征绝对数。如果是浮点数，代表考虑特征百分比，即考虑（百分比xN）取整后的特征数。其中N为样本总特征数。  一般来说，如果样本特征数不多，比如小于50，我们用默认的"`None`"就可以了，如果特征数非常多，我们可以灵活使用刚才描述的其他取值来控制划分时考虑的最大特征数，以控制决策树的生成时间。 |
| 决策树最大深度`max_depth`                          | 决策树的最大深度，默认可以不输入，如果不输入的话，决策树在建立子树的时候不会限制子树的深度。一般来说，数据少或者特征少的时候可以不管这个值。如果模型样本量多，特征也多的情况下，推荐限制这个最大深度，具体的取值取决于数据的分布。常用的可以取值10-100之间。 |
| 内部节点再划分所需最小样本数`min_samples_split`    | 这个值限制了子树继续划分的条件，如果某节点的样本数少于`min_samples_split`，则不会继续再尝试选择最优特征来进行划分， 默认是2，如果样本量不大，不需要管这个值。如果样本量数量级非常大，则推荐增大这个值。10万样本，建立决策树时，`min_samples_split`=10，可以作为参考。 |
| 叶子节点最少样本数`min_samples_leaf`               | 这个值限制了叶子节点最少的样本数，如果某叶子节点数目小于样本数，则会和兄弟节点一起被剪枝。 默认是1,可以输入最少的样本数的整数，或者最少样本数占样本总数的百分比。如果样本量不大，不需要管这个值。如果样本量数量级非常大，则推荐增大这个值。之前的10万样本项目使用`min_samples_leaf`的值为5，仅供参考。 |
| 叶子节点最小的样本权重和`min_weight_fraction_leaf` | 这个值限制了叶子节点所有样本权重和的最小值，如果小于这个值，则会和兄弟节点一起被剪枝。 默认是0，就是不考虑权重问题。一般来说，如果我们有较多样本有缺失值，或者分类树样本的分布类别偏差很大，就会引入样本权重，这时我们就要注意这个值了。 |
| 最大叶子节点数`max_leaf_nodes`                     | 通过限制最大叶子节点数，可以防止过拟合，默认是"`None`”，即不限制最大的叶子节点数。如果加了限制，算法会建立在最大叶子节点数内最优的决策树。如果特征不多，可以不考虑这个值，但是如果特征分成多的话，可以加以限制，具体的值可以通过交叉验证得到。 |
| 类别权重`class_weight`                             | 指定样本各类别的的权重，主要是为了防止训练集某些类别的样本过多，导致训练的决策树过于偏向这些类别。这里可以自己指定各个样本的权重，或者用“`balanced`”，如果使用“`balanced`”，则算法会自己计算权重，样本量少的类别所对应的样本权重会高。当然，如果你的样本类别分布没有明显的偏倚，则可以不管这个参数，选择默认的"`None`" |
| 节点划分最小不纯度`min_impurity_split`             | 这个值限制了决策树的增长，如果某节点的不纯度(基尼系数，信息增益，均方差，绝对差)小于这个阈值，则该节点不再生成子节点。即为叶子节点 。 |
| 数据是否预排序`presort`                            | 这个值是布尔值，默认是False不排序。一般来说，如果样本量少或者限制了一个深度很小的决策树，设置为true可以让划分点选择更加快，决策树建立的更加快。如果样本量太大的话，反而没有什么好处。问题是样本量少的时候，我速度本来就不慢。所以这个值一般懒得理它就可以了。 |

```python
def MakeTree_(inputfile):
    #为防止出现NaN值，已调用填充Nan值方法(没有Nan值也不影响)
    #想了半天还是自己去调参，调参部分已经标注
    #会在程序运行的目录下生成tree.dot
    return True
##输出模型评估报告+模型的预测准确率+决策树可视化+ROC曲线+P-R曲线+最大KS+最佳阈值+决策树可视化(这个东西比较大)
```



- [x] 已实现，调参自行调整，不附加默认值

## 4、交叉验证+网格搜索+随机搜索

### 4.1 交叉验证+网格搜索

#### 4.1.1 `scikit-learn` 中的交叉验证

```python
sklearn.model_selection.cross_val_score(estimator, X, y=None, *, groups=None, scoring=None, cv=None, n_jobs=None, verbose=0,fit_params=None, pre_dispatch='2\n_jobs', error_score=nan)
```

Parameters

- **estimator** estimator object implementing ‘fit’*

  The object to use to fit the data.

- **X** array-like of shape (n_samples, n_features)*

  The data to fit. Can be for example a list, or an array.

- **y** array-like of shape (n_samples,) or (n_samples, n_outputs), default=None*

  The target variable to try to predict in the case of supervised learning.

- **groups** array-like of shape (n_samples,), default=None*

  Group labels for the samples used while splitting the dataset into train/test set. Only used in conjunction with a “Group” [cv](https://scikit-learn.org/stable/glossary.html#term-cv) instance (e.g., [`GroupKFold`](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GroupKFold.html#sklearn.model_selection.GroupKFold)).

- **scoring** str or callable, default=None*

  A str (see model evaluation documentation) or a scorer callable object / function with signature `scorer(estimator, X, y)` which should return only a single value.Similar to [`cross_validate`](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.cross_validate.html#sklearn.model_selection.cross_validate) but only a single metric is permitted.If None, the estimator’s default scorer (if available) is used.

- **cv** int, cross-validation generator or an iterable, default=None*

  Determines the cross-validation splitting strategy. Possible inputs for cv are:None, to use the default 5-fold cross validation,int, to specify the number of folds in a `(Stratified)KFold`,[CV splitter](https://scikit-learn.org/stable/glossary.html#term-cv-splitter),An iterable yielding (train, test) splits as arrays of indices.For int/None inputs, if the estimator is a classifier and `y` is either binary or multiclass, [`StratifiedKFold`](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.StratifiedKFold.html#sklearn.model_selection.StratifiedKFold) is used. In all other cases, [`KFold`](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.KFold.html#sklearn.model_selection.KFold) is used.Refer [User Guide](https://scikit-learn.org/stable/modules/cross_validation.html#cross-validation) for the various cross-validation strategies that can be used here.*Changed in version 0.22:* `cv` default value if None changed from 3-fold to 5-fold.

- **n_jobs** int, default=None*

  The number of CPUs to use to do the computation. `None` means 1 unless in a [`joblib.parallel_backend`](https://joblib.readthedocs.io/en/latest/parallel.html#joblib.parallel_backend) context. `-1` means using all processors. See [Glossary](https://scikit-learn.org/stable/glossary.html#term-n-jobs) for more details.

- **verbose** int, default=0*

  The verbosity level.

- **fit_params** dict, default=None*

  Parameters to pass to the fit method of the estimator.

- **pre_dispatch** int or str, default=’2\*n_jobs’*

  Controls the number of jobs that get dispatched during parallel execution. Reducing this number can be useful to avoid an explosion of memory consumption when more jobs get dispatched than CPUs can process. This parameter can be:None, in which case all the jobs are immediately created and spawned. Use this for lightweight and fast-running jobs, to avoid delays due to on-demand spawning of the jobsAn int, giving the exact number of total jobs that are spawnedA str, giving an expression as a function of n_jobs, as in ‘2*n_jobs’

- **error_score** ‘raise’ or numeric, default=np.nan*

  Value to assign to the score if an error occurs in estimator fitting. If set to ‘raise’, the error is raised. If a numeric value is given, FitFailedWarning is raised. This parameter does not affect the refit step, which will always raise the error.*New in version 0.20.*

Returns

- **scores** array of float, shape=(len(list(cv)),)*

  Array of scores of the estimator for each run of the cross validation.

```python
def Simplecross_val_score(inputfile):#传入输入文件
    return True
#会出输出k={}，验证集上的准确率={:.3f}以及SVM 在不同的C值时，它在训练集和交叉验证上的分数的图形
#可自行调整Simplecross_val_score()中各种参数
```

- [x] 简单的网格搜索



#### 4.1.2 `k` 折交叉验证器

提供训练/测试索引以将数据拆分为训练/测试集。将数据集拆分为k个连续的折叠（默认情况下不进行混洗）。

然后将每个折叠用作一次验证，而剩余k-1个折叠形成训练集。

```python
class sklearn.model_selection.KFold(n_splits=5, *, shuffle=False, random_state=None)
```

参数

- **n_splits** *int，默认= 5*

  折数。必须至少为2。*在0.22版中更改：*`n_splits`默认值从3更改为5。

- **shuffle** *bool，默认为False*

  在拆分成批次之前是否对数据进行混洗。请注意，每个拆分内的样本都不会被混洗。

- **random_state** *int或RandomState实例，默认=无*

  当`shuffle`为True时，`random_state`会影响索引的顺序，从而控制每个折叠的随机性。否则，此参数无效。为多个函数调用传递可重复输出的int值。请参阅[词汇表](https://scikit-learn.org/stable/glossary.html#term-random-state)。

**本项目运用`sklearn`决策树分类器使用（网格搜索+交叉验证）**

```python
def Cross_Validation(inputfile,n):##传入输入文件，n为n_splits
    #上述系数若想修改去QuickInit中修改Cross_Validation()
    return True
#会生成最佳的max_depth和criterion+Best estimator在将此参数带入上面的构建决策树中
#其他参数规划正在调整
```

- [x] 完成

### 4.2 随机搜索

```python
sklearn.model_selection.RandomizedSearchCV(estimator, param_distributions, *, n_iter=10, scoring=None, n_jobs=None, iid='deprecated', refit=True, cv=None, verbose=0, pre_dispatch='2*n_jobs', random_state=None, error_score=nan, return_train_score=False)
```

Parameters

- **estimator***estimator object.*

  A object of that type is instantiated for each grid point. This is assumed to implement the scikit-learn estimator interface. Either estimator needs to provide a `score` function, or `scoring` must be passed.

- **param_distributions***dict or list of dicts*

  Dictionary with parameters names (`str`) as keys and distributions or lists of parameters to try. Distributions must provide a `rvs` method for sampling (such as those from scipy.stats.distributions). If a list is given, it is sampled uniformly. If a list of dicts is given, first a dict is sampled uniformly, and then a parameter is sampled using that dict as above.

- **n_iter***int, default=10*

  Number of parameter settings that are sampled. n_iter trades off runtime vs quality of the solution.

- **scoring***str, callable, list/tuple or dict, default=None*

  A single str (see [The scoring parameter: defining model evaluation rules](https://scikit-learn.org/stable/modules/model_evaluation.html#scoring-parameter)) or a callable (see [Defining your scoring strategy from metric functions](https://scikit-learn.org/stable/modules/model_evaluation.html#scoring)) to evaluate the predictions on the test set.For evaluating multiple metrics, either give a list of (unique) strings or a dict with names as keys and callables as values.NOTE that when using custom scorers, each scorer should return a single value. Metric functions returning a list/array of values can be wrapped into multiple scorers that return one value each.See [Specifying multiple metrics for evaluation](https://scikit-learn.org/stable/modules/grid_search.html#multimetric-grid-search) for an example.If None, the estimator’s score method is used.

- **n_jobs***int, default=None*

  Number of jobs to run in parallel. `None` means 1 unless in a [`joblib.parallel_backend`](https://joblib.readthedocs.io/en/latest/parallel.html#joblib.parallel_backend) context. `-1` means using all processors. See [Glossary](https://scikit-learn.org/stable/glossary.html#term-n-jobs) for more details.*Changed in version v0.20:* `n_jobs` default changed from 1 to None

- **pre_dispatch***int, or str, default=None*

  Controls the number of jobs that get dispatched during parallel execution. Reducing this number can be useful to avoid an explosion of memory consumption when more jobs get dispatched than CPUs can process. This parameter can be:None, in which case all the jobs are immediately created and spawned. Use this for lightweight and fast-running jobs, to avoid delays due to on-demand spawning of the jobsAn int, giving the exact number of total jobs that are spawnedA str, giving an expression as a function of n_jobs, as in ‘2*n_jobs’

- **iid***bool, default=False*

  If True, return the average score across folds, weighted by the number of samples in each test set. In this case, the data is assumed to be identically distributed across the folds, and the loss minimized is the total loss per sample, and not the mean loss across the folds.*Deprecated since version 0.22:* Parameter `iid` is deprecated in 0.22 and will be removed in 0.24

- **cv***int, cross-validation generator or an iterable, default=None*

  Determines the cross-validation splitting strategy. Possible inputs for cv are:None, to use the default 5-fold cross validation,integer, to specify the number of folds in a `(Stratified)KFold`,[CV splitter](https://scikit-learn.org/stable/glossary.html#term-cv-splitter),An iterable yielding (train, test) splits as arrays of indices.For integer/None inputs, if the estimator is a classifier and `y` is either binary or multiclass, [`StratifiedKFold`](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.StratifiedKFold.html#sklearn.model_selection.StratifiedKFold) is used. In all other cases, [`KFold`](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.KFold.html#sklearn.model_selection.KFold) is used.Refer [User Guide](https://scikit-learn.org/stable/modules/cross_validation.html#cross-validation) for the various cross-validation strategies that can be used here.*Changed in version 0.22:* `cv` default value if None changed from 3-fold to 5-fold.

- **refit***bool, str, or callable, default=True*

  Refit an estimator using the best found parameters on the whole dataset.For multiple metric evaluation, this needs to be a `str` denoting the scorer that would be used to find the best parameters for refitting the estimator at the end.Where there are considerations other than maximum score in choosing a best estimator, `refit` can be set to a function which returns the selected `best_index_` given the `cv_results`. In that case, the `best_estimator_` and `best_params_` will be set according to the returned `best_index_` while the `best_score_` attribute will not be available.The refitted estimator is made available at the `best_estimator_` attribute and permits using `predict` directly on this `RandomizedSearchCV` instance.Also for multiple metric evaluation, the attributes `best_index_`, `best_score_` and `best_params_` will only be available if `refit` is set and all of them will be determined w.r.t this specific scorer.See `scoring` parameter to know more about multiple metric evaluation.*Changed in version 0.20:* Support for callable added.

- **verbose***integer*

  Controls the verbosity: the higher, the more messages.

- **random_state***int or RandomState instance, default=None*

  Pseudo random number generator state used for random uniform sampling from lists of possible values instead of scipy.stats distributions. Pass an int for reproducible output across multiple function calls. See [Glossary](https://scikit-learn.org/stable/glossary.html#term-random-state).

- **error_score***‘raise’ or numeric, default=np.nan*

  Value to assign to the score if an error occurs in estimator fitting. If set to ‘raise’, the error is raised. If a numeric value is given, FitFailedWarning is raised. This parameter does not affect the refit step, which will always raise the error.

- **return_train_score***bool, default=False*

  If `False`, the `cv_results_` attribute will not include training scores. Computing training scores is used to get insights on how different parameter settings impact the overfitting/underfitting trade-off. However computing the scores on the training set can be computationally expensive and is not strictly required to select the parameters that yield the best generalization performance.*New in version 0.19.**Changed in version 0.21:* Default value was changed from `True` to `False`

```python
def Sample_RandomSearch(inputfile):
    #里面的参数全部可调
    #额外附加['xgboost']
    return True
#输出最优的训练器+最优训练器的精度->决策树
```



- [x] 实现

## 5、准确率+精确率+召回率F-score 给出分析

方法论准备写到QuickFun。~~在写了在写了~~

- [x] 实现

## 6、ROC+KS+Precision-Recall

方法论准备写到QuickFun。~~在写了在写了~~

- [x] 实现

# 项目实现

```python
def 在写了在写了()：
	return True
```

