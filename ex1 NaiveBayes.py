"""
《机器学习》作业

学号：2000900
姓名：张敬川
11.19

调试记录：

"""
import numpy as np
from sklearn import datasets

# 定义朴素贝叶斯类
class NaiveBayes(object):

    def __init__(self, x, y):
        self.x = x  # 决策变量
        self.y = y  # 类别
        self.set_model_para()  # 设置其他参数

    # 设置模型的参数
    def set_model_para(self):

        self.m = self.x.shape[0]  # 样本个数
        self.n = self.x.shape[1]  # 特征个数

        self.labels = []  # 标签库
        for i in range(self.m):
            # 若该标签不在标签库中
            if np.isin(self.y[i], self.labels, invert=True):
                self.labels.append(self.y[i])  # 添加该标签到标签库

        self.class_num = len(self.labels)  # 类别数

    # 找到各个类别所包括的样本编号
    def find_index(self):
        index = list(np.zeros([self.class_num]))
        for i in range(self.class_num):
            index[i] = []

        for i in range(self.m):
            for j in range(self.class_num):
                if self.y[i] == self.labels[j]:
                    index[j].append(i)
                    break
        return index

    # 计算先验概率: P(A)
    def compute_PA(self):
        PA = np.zeros([self.class_num])  # 初始化
        for i in range(self.class_num):
            PA[i] = np.mean(np.equal(self.y, self.labels[i]))
        return PA

    # 计算概率密度函数: P(B1|A), P(B2|A), ... , P(Bn|A), 其实是计算μ和σ
    def compute_ProbDensityFun(self):
        # 获取每个类别所含样本的编号
        index = self.find_index()
        # 初始化矩阵
        mu = np.zeros([self.class_num, self.n])     # 储存均值
        sigma = np.zeros([self.class_num, self.n])  # 储存标准差

        for i in range(self.class_num):
            for j in range(self.n):
                data_x = self.x[index[i], j]  # 找到y(i)类别下的x(j)列数据
                mu[i, j] = np.mean(data_x)    # 求均值μ
                sigma[i, j] = np.std(data_x)  # 求标准差σ

        return mu, sigma

    # 训练
    def fit(self):
        # 计算先验概率
        self.PA = self.compute_PA()
        # 计算概率密度函数的均值和方差
        self.mu, self.sigma = self.compute_ProbDensityFun()

    # 测试
    def predict(self, input_x):

        # 根据mu、sigma、变量值来计算对应的正态分布概率
        def norm(mu, sigma, x):
            res = 1/(np.sqrt(2*np.pi)*sigma) * \
                  np.exp(-(x-mu)**2/(2*sigma**2))
            return res

        m = input_x.shape[0]       # 预测样本的个数
        n = input_x.shape[1]       # 预测样本的维数
        self.pred = np.zeros([m])  # 储存预测结果

        for i in range(m):
            # 初始化概率
            self.prob = np.zeros([self.class_num])

            for j in range(self.class_num):
                self.PBA = 1  # 初始化

                for k in range(n):
                    # 计算P(B1|A)*P(B2|A)*...*P(Bn|A)
                    self.PBA *= norm(self.mu[j, k], self.sigma[j, k], input_x[i, k])

                # 计算分子P(B|A)P(A)
                self.prob[j] = self.PBA * self.PA[j]

            self.pred[i] = np.argmax(self.prob)  # 取最大值

        return self.pred


if __name__ == '__main__':
    # 导入数据
    iris = datasets.load_iris()
    # 实例化类
    model = NaiveBayes(iris.data, iris.target)
    # 训练（用全部150个样本）
    model.fit()
    # 测试（用全部150个样本）
    pred = model.predict(iris.data)
    # 计算准确率
    accuracy = np.mean(np.equal(pred, iris.target))
    print('预测准确率：%.2f%s' % (100*accuracy, '%'))




