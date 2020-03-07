import numpy as np
import math


def lp_distance(point1, point2, p=2):
    if len(point1) == len(point2) and len(point1)>1:
        sum_ = 0
        for i in range(len(point1)):
            sum_ += math.pow(abs(point1[i]-point2[i]), p)
        return math.pow(sum_, 1/p)
    else:
        return None


class Node:
    def __init__(self, data, depth=0, lchild=None, rchild=None):
        self.data = data
        self.lchild = lchild
        self.rchild = rchild
        self.depth = depth


class KdTree:
    def __init__(self):
        self.nearest = None
        self.root = None
        self.dimention = None
        self.train_data = None

    def create(self, data, depth=0):
        if len(data) == 0:
            return
        # 以axis索引的轴对多维数组进行排序
        size, m = np.shape(data)
        self.dimention = m-1
        axis = depth % self.dimention
        index = np.lexsort([data[:, axis]])
        sorted = data[index, :]
        # 取排序后的中位数，并作为当前结点
        mid = int(size / 2)
        node = Node(sorted[mid], depth)
        if depth==0:
            self.root = node
        # 左右子节点递归
        left = sorted[:mid]
        right = sorted[mid+1:]
        depth+=1
        node.lchild = self.create(left, depth)
        node.rchild = self.create(right, depth)
        return node

    def show(self, node):
        if node is not None:
            print(node.depth, node.data)
            self.show(node.lchild)
            self.show(node.rchild)

    def search(self, x, count=1):
        nearest = [] #[[dist, node]]
        for i in range(count):
            nearest.append([float("inf"), None])
        self.nearest = np.array(nearest)
        if self.dimention is None:
            return

        def recurve(node):
            if node is None:
                return

            axis = node.depth % self.dimention
            daxis = x[axis] - node.data[axis]
            if x[axis] < node.data[axis]:
                recurve(node.lchild)
            else:
                recurve(node.rchild)

            dist = lp_distance(x, node.data[:-1])

            if dist < np.max(self.nearest[:, 0]):
                index = np.where(self.nearest == np.max(self.nearest[:, 0]))
                self.nearest[index[0]] = [dist, node]

            if abs(daxis) < np.max(self.nearest[:, 0]) :
                # 需要检查另一区域
                if daxis > 0:
                    recurve(node.lchild)
                else:
                    recurve(node.rchild)

        recurve(self.root)
        #
        # knn = self.nearest[:, 1]
        # belong = []
        # for i in knn:
        #     belong.append(i.data[-1])
        # b = max(set(belong), key=belong.count)  # 注意这个用法
        #
        # return self.nearest, b


