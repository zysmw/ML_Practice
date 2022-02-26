# scikit-learn决策树算法类库内部实现是使用了调优过的CART树算法，既可以做分类，又可以做回归。
# 分类决策树的类对应的是DecisionTreeClassifier
# 回归决策树的类对应的是DecisionTreeRegressor
# 两者的参数定义几乎完全相同，但是意义不全相同
# 重点比较两者参数使用的不同点和调参的注意点

import sklearn
from sklearn import tree
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.datasets import load_iris
import sys
import os

print(sklearn.__version__)

# 可以发现DecisionTreeClassifier和DecisionTreeRegressor的参数设置大多一样
# 同时剪枝的操作已经通过参数给出来了，可以通过参数的调整来剪枝
# 具体在优化调参时可以再参照下刘建平的博客

DTC = DecisionTreeClassifier(criterion="gini", splitter="best", max_features=None, max_depth=None,
                             min_samples_split=2, min_samples_leaf=1, min_weight_fraction_leaf=0,
                             max_leaf_nodes=None, min_impurity_split=None,
                             class_weight=None)

DTR = DecisionTreeRegressor(criterion="mse", splitter="best", max_features=None, max_depth=None,
                            min_samples_split=2, min_samples_leaf=1, min_weight_fraction_leaf=0,
                            max_leaf_nodes=None, min_impurity_split=None)

# 注意：
# 1. 当样本数量少但是样本特征非常多的时候，决策树很容易过拟合，一般来说，样本数比特征数多一些会比较容易建立健壮的模型
# 2. 如果样本数量少但是样本特征非常多，在拟合决策树模型前，推荐先做维度规约，比如主成分分析（PCA），特征选择（Losso）或者独立成分分析（ICA）
# 3. 推荐多用决策树的可视化，同时先限制决策树的深度（比如最多3层），这样可以先观察下生成的决策树里数据的初步拟合情况，然后再决定是否要增加深度
# 4. 对于分类任务，训练模型前先观察样本类别分布情况，看是否不均匀，考虑用不用class_weight
# 5. 决策树的数组使用的是numpy的float32类型，如果训练数据不是这样的格式，算法会先做copy再运行
# 6. 如果输入的样本矩阵是稀疏的，推荐在拟合前调用 csc_matrix 稀疏化，在预测前调用 csr_matrix 稀疏化



# 现在来演示决策树可视化

os.environ["PATH"] += os.pathsep + "C:\\Program Files\\Graphviz\\bin"

iris = load_iris()
DTC = DTC.fit(iris.data, iris.target)
with open("iris.dot", "w") as f:
    f = tree.export_graphviz(DTC, out_file=f)
# 此时我们已经训练好了模型，并将模型保存在 ".dot"文件内
# 这时候我们有3种可视化方法

# 第一种方法
# 用graphviz的dot命令生成决策树的可视化文件，敲完这个命令后当前目录就可以看到决策树的可视化文件iris.jpg,打开可以看到决策树的模型图
# 在terminal或终端输入命令  dot -Tjpg -o iris.jpg iris.dot 就可以了，如果想要输出PDF文件，就把jpg都改成pdf即可，但pdf文件太慢了
# 所以说这一种方法适合输出图片来查看，如果想要得到PDF文件，用方法二

# 第二种方法
# 调用 pydotplus 模块，这样就不用命令行了
import pydotplus
dot_data = tree.export_graphviz(DTC, out_file=None,
                                feature_names=iris.feature_names,
                                class_names=iris.target_names,
                                filled=True, rounded=True,
                                special_characters=True)
graph = pydotplus.graph_from_dot_data(dot_data)
graph.write_pdf("iris_from_pydotplus.pdf")
graph.write_jpg("iris_from_pydotplus.jpg")
graph.write_png("iris_from_pydotplus.png")

# 第三种方法
# 可以直接把图产生在ipython的notebook
from IPython.display import Image
dot_data = tree.export_graphviz(DTC, out_file=None,
                                feature_names=iris.feature_names,
                                class_names=iris.target_names,
                                filled=True, rounded=True,
                                special_characters=True)  # 加上这些可以让图片效果更好
graph = pydotplus.graph_from_dot_data(dot_data)
Image(graph.create_png)

