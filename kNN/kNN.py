import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from kdTree import KdTree
import math

iris = load_iris();
df = pd.DataFrame(iris.data, columns=iris.feature_names)
df['label'] = iris.target
dataSet = np.array(df.iloc[:100,[0,1,-1]])
train, test = train_test_split(dataSet, test_size=0.4)
x0= np.array([x0 for i, x0 in enumerate(train) if train[i][-1]==0])
x1= np.array([x1 for i, x1 in enumerate(train) if train[i][-1]==1])


def lp_distance(point1, point2, p=2):
    if len(point1) == len(point2) and len(point1)>1:
        sum_ = 0
        for i in range(len(point1)):
            sum_ += math.pow(abs(point1[i]-point2[i]), p)
        return math.pow(sum_, 1/p)
    else:
        return None

# x_train_data = train[:,:2]
# x_train_label = train[:,-1]

model = KdTree()
model.create(train)
# model.show(model.root)
x= np.array([5.5, 3.6])
model.search(x, 2)
for i in model.nearest:
    print(i[1].data)


# def show_train():
#     plt.scatter(x0[:, 0], x0[:, 1], c='pink', label='[0]')
#     plt.scatter(x1[:, 0], x1[:, 1], c='orange', label='[1]')
#     plt.xlabel('sepal length')
#     plt.ylabel('sepal width')
#
# score = 0
# for x in test:
#     show_train()
#     plt.scatter(x[0], x[1], c='red', marker='x', label='test point')  # 测试点
#     near, belong = kdt.search(x[:-1], 5)  # 设置临近点的个数
#     if belong == x[-1]:
#         score += 1
#     print("test:")
#     print(x, "predict:", belong)
#     print("nearest:")
#     for n in near:
#         print(n[1].data, "dist:", n[0])
#         plt.scatter(n[1].data[0], n[1].data[1], c='green', marker='+')  # k个最近邻点
#     plt.legend()
#     plt.show()





