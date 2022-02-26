from sklearn.datasets import make_blobs
from numpy import  random, where
from sklearn.ensemble import IsolationForest
import matplotlib.pyplot as plt
import pandas as pd

random.seed(3)
X, _ = make_blobs(n_samples=300, centers=1, cluster_std=.3, center_box=(20, 5))
plt.scatter(X[:, 0], X[:, 1], marker="o", c=_, s=25, edgecolor="k")
plt.show()

IF = IsolationForest(n_estimators=100, contamination=.03)
predictions = IF.fit_predict(X)

outlier_index = where(predictions == -1)
values = X[outlier_index]
plt.scatter(X[:, 0], X[:, 1])
plt.scatter(values[:, 0], values[:, 1], color='y')
plt.show()

#########################################################################
file_path = "./data/data_GBDT.csv"
data = pd.read_csv(file_path)

target = "Disbursed"
IDcol = "ID"
feature_cols = [x for x in data.columns if x not in [target, IDcol]]
features = data[feature_cols]
labels = data[target]
print(features.shape)  # (20000, 49)

detections = IF.fit_predict(features.values)  # 注意这里尽量把dataframe转换为ndarray
print(detections.shape)  # (20000, )
print(detections)

outlier_index = where(detections == -1)[0]  # 注意这里返回的是一个元组，需要提取它的第一个元素才是index
print(len(outlier_index))  # 一共有600个异常值

outliers = features.iloc[outlier_index, :]
print(outliers)
