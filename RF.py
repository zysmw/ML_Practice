# 和GBDT的调参类似，RF需要调参的参数也包括两部分，第一部分是Bagging框架的参数，第二部分是CART决策树的参数。

# bagging框架参数
# 1. n_estimator: 默认是100
# 2. oob_score: 即是否采用袋外样本来评估模型的好坏。默认识False。个人推荐设置为True，因为袋外分数反应了一个模型拟合后的泛化能力。
# 3. criterion: 即CART树做划分时对特征的评价标准。分类模型和回归模型的损失函数是不一样的。
# 分类RF对应的CART分类树默认是基尼系数gini,另一个可选择的标准是信息增益。
# 回归RF对应的CART回归树默认是均方差mse，另一个可以选择的标准是绝对值差mae。
# 一般来说，默认即可

# 弱分类器参数
# 决策树参数中最重要的包括：
# 1. 最大特征数max_features
# 2. 最大深度max_depth
# 3. 内部节点再划分所需最小样本数min_samples_split
# 4. 叶子节点最少样本数min_samples_leaf

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, cross_validate
from sklearn import metrics

file_path = "./data/data_GBDT.csv"
data = pd.read_csv(file_path)

target = "Disbursed"
IDcol = "ID"

# print(data[target].value_counts())
# 数据极度不平衡

feature_cols = [x for x in data.columns if x not in [target, IDcol]]
features = data[feature_cols]
labels = data[target]

# 不管任何参数，都使用默认值，看下拟合效果
rf0 = RandomForestClassifier(oob_score=True, random_state=10)
rf0.fit(features, labels)
predict_prob0 = rf0.predict_proba(features)[:, 1]
print("全参数默认模型表现")
print(f"OOB Score: {rf0.oob_score_}")
print(f"AUC Score: {metrics.roc_auc_score(labels, predict_prob0)}")

# 对n_estimators进行网格搜索
param_list1 = {"n_estimators": range(10, 91, 10)}
base_estimator_1 = RandomForestClassifier(min_samples_split=100, min_samples_leaf=20,
                                          max_depth=8, max_features="auto",
                                          random_state=10)
gs1 = GridSearchCV(estimator=base_estimator_1, param_grid=param_list1, scoring="roc_auc", cv=5, n_jobs=-1)
gs1.fit(features, labels)
print("\n第一次搜索：弱学习器个数 n_estimators")
print(f"最佳结果：{gs1.best_params_}")
print(f"最佳AUC分数：{gs1.best_score_}")
# 确定了最佳弱学习器个数：60


# 对max_depth和min_samples_split进行最佳搜索
param_list2 = {"max_depth": range(3, 14, 2), "min_samples_split": range(50, 201, 20)}
base_estimator_2 = RandomForestClassifier(n_estimators=60, min_samples_leaf=20, max_features="auto",
                                          oob_score=True, random_state=10)
gs2 = GridSearchCV(estimator=base_estimator_2, param_grid=param_list2, scoring="roc_auc", cv=5, n_jobs=-1)
gs2.fit(features, labels)
print("\n第二次搜索：树的最大深度max_depth 以及 节点最小样本划分个数min_samples_split")
print(f"最佳结果：{gs2.best_params_}")
print(f"最佳AUC分数：{gs2.best_score_}")
# 找到最优结果 max_depth = 13   min_samples_split = 110


# 综合上述的最优参数再次创建RF模型
rf1 = RandomForestClassifier(n_estimators=60, min_samples_split=110, min_samples_leaf=20, max_depth=13,
                             max_features="auto", oob_score=True, random_state=10)
rf1.fit(features, labels)
predict_prob1 = rf1.predict_proba(features)[:, 1]
print("\n参数调优后模型表现")
print(f"OOB Score: {rf1.oob_score_}")
print(f"AUC Score: {metrics.roc_auc_score(labels, predict_prob1)}")
# 可以发现我们的OOB 分数有所提高，也就是说模型的泛化能力增强了


# 需要注意的是，对于min_samples_leaf和min_samples_split这两个参数，我们一般要一起调参，而不是分开调
param_list3 = {"min_samples_leaf": range(10, 61, 10), "min_samples_split": range(80, 161, 20)}
base_estimator_3 = RandomForestClassifier(n_estimators=60, max_depth=13,
                                          max_features="auto", oob_score=True, random_state=10)
gs3 = GridSearchCV(estimator=base_estimator_3, param_grid=param_list3, scoring="roc_auc", n_jobs=-1, cv=5)
gs3.fit(features, labels)
print("\n第三次搜索：节点最小样本个数min_samples_leaf 以及 节点最小样本划分个数min_samples_split")
print(f"最佳结果：{gs3.best_params_}")
print(f"最佳AUC分数：{gs3.best_score_}")
# 得到最优参数  min_samples_split = 120   min_samples_leaf = 20


# 最后我们再对最大特征数max_features进行调参
# 由于我们的总特征数为50，之前一直采用的是平方根特征数，本次调参我们具体到数值
param_list4 = {"max_features": range(3, 12, 2)}
base_estimator_4 = RandomForestClassifier(n_estimators=60, max_depth=13,
                                          min_samples_leaf=20, min_samples_split=120,
                                          oob_score=True, random_state=10)
gs4 = GridSearchCV(estimator=base_estimator_4, param_grid=param_list4, scoring="roc_auc", n_jobs=-1, cv=5)
gs4.fit(features, labels)
print("\n第四次搜索：最大特征数 max_features")
print(f"最佳结果：{gs4.best_params_}")
print(f"最佳AUC分数：{gs4.best_score_}")
# 得到最优参数 max_features = 7


# 此时我们综合上述所有的最佳参数来构建我们最终的模型
rf = RandomForestClassifier(n_estimators=60, max_depth=13, max_features=7,
                            min_samples_leaf=20, min_samples_split=120,
                            oob_score=True, random_state=10)
rf.fit(features, labels)
print(f"\n模型最终OOB分数： {rf.oob_score_}")
