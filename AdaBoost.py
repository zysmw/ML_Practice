# scikit-learn中Adaboost类库比较直接，就是AdaBoostClassifier和AdaBoostRegressor两个
# 从名字就能看出它们二者的区别

# AdaBoostClassifier使用了两种Adaboost分类算法的实现，SAMME和SAMME.R
# AdaBoostRegressor则使用了Adaboost回归算法的实现，即Adaboost.R2

# 当我们对Adaboost调参时，主要要对两部分内容进行调参
# 第一部分是对我们的Adaboost的框架进行调参
# 第二部分是对我们选择的弱分类器进行调参
# 两者相辅相成

# 我们首先来看AdaBoostClassifier和AdaBoostRegressor框架参数，重要参数如下：
# 1. base_estimator
# AdaBoostClassifier和AdaBoostRegressor都有，即我们的弱分类学习器或者弱回归学习器
# 理论上可以选择任何一个分类或者回归学习器，不过需要支持样本权重,我们常用的一般是CART决策树或者神经网络MLP。默认是决策树

# 2. algorithm
# 这个参数只有AdaBoostClassifier有。主要原因是scikit-learn实现了两种Adaboost分类算法，SAMME和SAMME.R
# 两者的主要区别是弱学习器权重的度量
# SAMME使用了二元分类Adaboost算法的扩展，即用对样本集分类效果作为弱学习器权重
# SAMME.R使用了对样本集分类的预测概率大小来作为弱学习器权重
# 由于SAMME.R使用了概率度量的连续值，迭代一般比SAMME快，因此AdaBoostClassifier的默认算法algorithm的值也是SAMME.R，我们一般使用默认的SAMME.R就够了
# 但是要注意的是使用了SAMME.R， 则弱分类学习器参数base_estimator必须限制使用支持概率预测的分类器。SAMME算法则没有这个限制

# 3. loss
# 这个参数只有AdaBoostRegressor有，Adaboost.R2算法需要用到
# 有线性‘linear’, 平方‘square’和指数 ‘exponential’三种选择, 默认是线性，一般使用线性就足够了，除非你怀疑这个参数导致拟合程度不好

# 4. n_estimator
# AdaBoostClassifier和AdaBoostRegressor都有，就是我们的弱学习器的最大迭代次数，或者说最大的弱学习器的个数
# 过多会过拟合，过少会欠拟合，默认是50
# 实际调参过程中，我们常常将n_estimator和learning_rate一起考虑

# 5. learning_rate
#  AdaBoostClassifier和AdaBoostRegressor都有，即每个弱学习器的权重缩减系数𝜈
# 较小的𝜈意味着我们需要更多的弱学习器的迭代次数
# 一般来说，可以从一个小一点的𝜈开始调参，默认是1

# 弱学习器参数，参考CART决策树的参数设定

import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import make_gaussian_quantiles

# 生成2维正态分布，生成的数据按分位数分为两类，500个样本,2个样本特征，协方差系数为2
X1, y1 = make_gaussian_quantiles(cov=2.0, n_samples=500, n_features=2, n_classes=2, random_state=1)

X2, y2 = make_gaussian_quantiles(mean=(3, 3), cov=1.5, n_samples=400, n_features=2, n_classes=2, random_state=1)

# 将两组数据合并为一组
X = np.concatenate((X1, X2))
y = np.concatenate((y1, -y2 + 1))

plt.scatter(X[:, 0], X[:, 1], marker="o", c=y)
plt.show()

# 使用基于决策树的Adaboost来做分类
# 这里我们选择了SAMME算法，最多200个弱分类器，步长0.8
bdt1 = AdaBoostClassifier(base_estimator=DecisionTreeClassifier(max_depth=2, min_samples_split=20, min_samples_leaf=5),
                          algorithm="SAMME",
                          n_estimators=200, learning_rate=0.8)
bdt1.fit(X, y)

# 模型拟合完毕，我们用网格图来查看拟合的区域
x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.02),
                     np.arange(y_min, y_max, 0.02))

Z = bdt1.predict(np.c_[xx.ravel(), yy.ravel()])
# 注意这里，直接调用np.reshape函数，而不要Z.reshape()
zz = np.reshape(Z, xx.shape)

cs = plt.contourf(xx, yy, zz)
plt.scatter(X[:, 0], X[:, 1], marker="o", c=y)
plt.show()

# 查看准确率
print(f"1st Score: {bdt1.score(X, y)}")

# 现在我们将弱分类器个数提升到300
bdt2 = AdaBoostClassifier(base_estimator=DecisionTreeClassifier(max_depth=2, min_samples_split=20, min_samples_leaf=5),
                          algorithm="SAMME",
                          n_estimators=300, learning_rate=0.8)
bdt2.fit(X, y)
print(f"2nd Score: {bdt2.score(X, y)}")
# 这印证了我们前面讲的，弱分离器个数越多，则拟合程度越好，当然也越容易过拟合

# 现在我们降低步长，即减少learning rate
bdt3 = AdaBoostClassifier(base_estimator=DecisionTreeClassifier(max_depth=2, min_samples_split=20, min_samples_leaf=5),
                          algorithm="SAMME",
                          n_estimators=300, learning_rate=0.5)
bdt3.fit(X, y)
print(f"3rd Score: {bdt3.score(X, y)}")
# 可见在同样的弱分类器的个数情况下，如果减少步长，拟合效果会下降

# 最后，我们将弱分类器个数设为700， 步长为0.7
bdt4 = AdaBoostClassifier(base_estimator=DecisionTreeClassifier(max_depth=2, min_samples_split=20, min_samples_leaf=5),
                          algorithm="SAMME",
                          n_estimators=700, learning_rate=0.7)
bdt4.fit(X, y)
print(f"4th Score: {bdt4.score(X, y)}")
# 此时的拟合分数和我们最初的300弱分类器，0.8步长的拟合程度相当。
# 也就是说，在我们这个例子中，如果步长从0.8降到0.7，则弱分类器个数要从300增加到700才能达到类似的拟合效果。
# 可见，弱分类器个数和步长需要同步进行调节
