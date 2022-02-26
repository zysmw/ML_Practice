# 梯度提升树GBDT
# GradientBoostingClassifier为GBDT的分类类， 而GradientBoostingRegressor为GBDT的回归类。两者的参数类型完全相同
# 在这些参数中，同AdaBoost一样，分为两个部分，一是Boosting框架的参数，二是弱学习器CART回归树的参数

# boosting框架参数
# 大部分都和Adaboost一样
# 1. n_estimator
# 2. learning_rate
# 3. subsample: 这是正则化的子采样，取值范围(0, 1]，注意这个采样是不放回的！推荐取值范围[0.5, 0.8]，系统默认1.0，即不使用子采样
# 4. init：即初始化的弱学习器，一般不用管
# 5. loss：损失函数，分类和回归是不一样的，这里需要着重区分：

# 对于分类模型，有对数似然损失函数"deviance"和指数损失函数"exponential"两者输入选择。默认是对数似然损失函数"deviance"。
# 一般来说，分类模型推荐使用默认的"deviance"，它对二元分离和多元分类各自都有比较好的优化。而指数损失函数等于把我们带到了Adaboost算法

# 对于回归模型，有均方差"ls", 绝对损失"lad", Huber损失"huber"和分位数损失“quantile”
# 默认是均方差"ls"
# 一般来说，如果数据的噪音点不多，用默认的均方差"ls"比较好
# 如果是噪音点较多，则推荐用抗噪音的损失函数"huber"。而如果我们需要对训练集进行分段预测的时候，则采用“quantile”

# 6. alpha：这个参数只有GradientBoostingRegressor有
# 当我们使用Huber损失"huber"和分位数损失“quantile”时，需要指定分位数的值。默认是0.9，如果噪音点较多，可以适当降低这个分位数的值。

# GBDT弱学习器参数
# 参照决策树的参数设置

# 下面以一个二元分类问题来实践GBDT调参

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import GradientBoostingClassifier
from sklearn import metrics
from sklearn.model_selection import cross_val_score, cross_validate
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV

file_path = "./data/data_GBDT.csv"
train = pd.read_csv(file_path)
# print(train.head(2))
# print(train.columns)
# 原始数据一共有51列，其中 Disbursed 列是我们的target，是一个只取0和1的二元target

target = "Disbursed"
IDcol = "ID"  # ID 这一列不作为特征
# print(train[target].value_counts())
# 19680个0,320个1，可见类别分布不平衡

x_columns = [x for x in train.columns if x not in [target, IDcol]]  # 选取特征列
X = train[x_columns]  # 特征
Y = train[target]  # 类别

# 首先我们不进行任何参数调整，均使用默认的，来看下拟合效果
gbm0 = GradientBoostingClassifier(random_state=0)
gbm0.fit(X, Y)
y_pred = gbm0.predict(X)
y_predprob = gbm0.predict_proba(X)[:, 1]
print(f"Accuracy: {metrics.accuracy_score(Y.values, y_pred)}")
print(f"AUC Score: {metrics.roc_auc_score(Y, y_predprob)}")
# Accuracy: 0.98525
# AUC Score: 0.9005309165396341
# 可见拟合的还算可以，下面我们通过调参来提高模型泛化能力


# 首先我们从步长learning rate 和迭代次数n_estimator入手
# 一般来说，开始选择一个较小的步长来网格搜索最好的迭代次数，我们先将步长设置为0.1
param_test1 = {"n_estimators": np.arange(20, 81, 10)}
gridSearch1 = GridSearchCV(estimator=GradientBoostingClassifier(learning_rate=0.1, min_samples_split=300,
                                                                min_samples_leaf=20, max_depth=8, max_features="sqrt",
                                                                subsample=0.8, random_state=10),
                           param_grid=param_test1, scoring="roc_auc", cv=5)
gridSearch1.fit(X, Y)
print(gridSearch1.cv_results_)  # 给出不同参数情况下的评价结果的记录
print(gridSearch1.best_params_)  # 描述了已取得最佳结果的参数的组合
print(gridSearch1.best_score_)  # 提供优化过程期间观察到的最好的评分
# 最好结果是 n_estimators = 20


# 找到了一个合适的迭代次数，现在我们开始对决策树进行调参。
# 首先我们对决策树最大深度max_depth和内部节点再划分所需最小样本数min_samples_split进行网格搜索。
param_test2 = {"max_depth": range(3, 14, 2), "min_samples_split": range(100, 801, 200)}
gridSearch2 = GridSearchCV(estimator=GradientBoostingClassifier(learning_rate=0.1, n_estimators=60, min_samples_leaf=20,
                                                                max_features="sqrt", subsample=0.8, random_state=10),
                           param_grid=param_test2, scoring="roc_auc", cv=5)
gridSearch2.fit(X, Y)
print(gridSearch2.best_params_)
print(gridSearch2.best_score_)
# {'max_depth': 5, 'min_samples_split': 300}

# 由于决策树深度5是一个比较合理的值，我们把它定下来
# 对于内部节点再划分所需最小样本数min_samples_split，我们暂时不能一起定下来，因为这个还和决策树其他的参数存在关联。
# 下面我们再对内部节点再划分所需最小样本数min_samples_split和叶子节点最少样本数min_samples_leaf一起调参
param_test3 = {'min_samples_split': range(800, 1900, 200), 'min_samples_leaf': range(60, 101, 10)}
gridSearch3 = GridSearchCV(estimator=GradientBoostingClassifier(learning_rate=0.1, n_estimators=60, max_depth=5,
                                                                max_features="sqrt", subsample=0.8, random_state=10),
                           param_grid=param_test3, scoring="roc_auc", n_jobs=-1, cv=5)
gridSearch3.fit(X, Y)
print(gridSearch3.best_params_)
print(gridSearch3.best_score_)
# {'min_samples_leaf': 100, 'min_samples_split': 800}
# 上面是输出结果，我们可以发现两个参数的最优值都取在了边界处，这就告诉我们可以进一步扩大或缩小边界来进行搜索

# 我们这里只做三次搜索，重点是体会整个调参过程，关键就是“控制变量”，以及一次只选择少量几个参数进行搜索调整，逐步确定逐步完善

# 后续我们还可以进一步对其它参数进行搜索，比如max_features，subsample等等

# 在我们确定了其它参数后我们就可以着手对learning_rate来进行实验调参了，因为上述我们所有的搜索过程都是基于一个恒定的步长

# 同时最后的评估标准也并不一定说AUC值越高越好，因为还要考虑到泛化能力
