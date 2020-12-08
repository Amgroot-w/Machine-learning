"""
《机器学习》作业 --- 2.Logistic回归

学号：2000900
姓名：张敬川
11.30

调试记录：

"""
# %% 导入包
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cap
# 显示中文
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# %% 定义Logistic回归类
class logistic(object):

    def __init__(self, x, y):
        self.x = x  # x: 特征x：m×(1+n)
        self.y = y  # 标签y：m×1，标签取值为0和1，二分类
        # 输入矩阵x必须满足：①已经归一化；②第一列为1（截距项）
        # 输出参数theta的第一个数为theta0（截距项系数）

    def train(self, epochs=10000, alpha=0.05, lamda=20):
        self.epochs = epochs  # 迭代次数
        self.alpha = alpha    # 学习率
        self.lamda = lamda    # 正则化参数

        m = self.x.shape[0]  # 样本数
        n = self.x.shape[1]  # 特征数
        self.theta = np.random.uniform(-1, 1, [self.x.shape[1], 1])  # 参数初始化
        delta = np.zeros([n, 1])  # 梯度初始化
        cost_history = {'epoch': [], 'cost': []}  # 字典记录误差变化
        # 训练
        for epoch in range(self.epochs):
            # 假设函数h(θ)
            h = cap.sigmoid(np.matmul(self.x, self.theta))
            # 交叉熵损失 + 正则化项
            J = cap.cross_entropy(h, self.y) + self.lamda * 1/(2*m) * np.sum(pow(self.theta[1:n, :], 2))
            # 计算梯度
            delta[0, :] = 1/m * np.matmul(self.x.T[0, :], h-self.y)  # theta0不加正则化
            delta[1:n, :] = 1/m * np.matmul(self.x.T[1:n, :], h-self.y) + self.lamda*1/m*self.theta[1:n, :]
            # 参数更新
            self.theta = self.theta - self.alpha * delta
            # 记录误差cost
            cost_history['epoch'].append(epoch)
            cost_history['cost'].append(J)
        # 返回值：参数theta((1+n)×1)
        return self.theta

    # 预测
    def predict(self, x, show=True):
        predx = np.matmul(x, self.theta)
        self.pred = np.array([1 if predx[i] > 0.5 else 0 for i in range(x.shape[0])]).reshape(-1, 1)
        if show:
            ac = np.mean(np.equal(self.pred, self.y))
            print("准确率：%.2f%s" % (ac*100, '%'))
        return self.pred

    # 绘制原始样本分布
    @staticmethod
    def plot_original_data(data):
        colors = ['c', 'orange']
        marker = ['o', 's']
        for i in range(2):
            score1 = data.loc[data['result'] == i]['score1']
            score2 = data.loc[data['result'] == i]['score2']
            plt.scatter(score1, score2, c=colors[i], marker=marker[i], s=50, linewidths=0.8, edgecolors='k')

        plt.xlabel('x1')
        plt.ylabel('x2')
        plt.title('原始样本分布')
        plt.show()

    # 绘制决策边界
    def plot_decision_boundary(self, data):
        X = data.values
        x_min, x_max = X[:, 0].min() - .5, X[:, 0].max() + .5
        y_min, y_max = X[:, 1].min() - .5, X[:, 1].max() + .5
        h = 0.05
        xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
        Z = self.predict(cap.feature_mapping(xx.ravel(), yy.ravel(), degree=8), show=False)
        Z = Z.reshape(xx.shape)
        plt.contourf(xx, yy, Z, cmap=plt.cm.Spectral)  # 决策边界

        # 原始样本分布图
        colors = ['c', 'orange']
        marker = ['o', 's']
        for i in range(2):
            score1 = data.loc[data['result'] == i]['score1']
            score2 = data.loc[data['result'] == i]['score2']
            plt.scatter(score1, score2, c=colors[i], marker=marker[i], s=50, linewidths=0.8, edgecolors='k')

        plt.xlabel('x1')
        plt.ylabel('x2')
        plt.title('Logistic回归后绘制出的决策边界')
        plt.show()


# %% 在数据集（线性不可分）测试模型
if __name__ == '__main__':
    data = pd.read_csv('data_Logistic.csv', names=['score1', 'score2', 'result'])                 # 导入数据
    data.iloc[:, :2] = (data.iloc[:, :2] - np.mean(data.iloc[:, :2])) / np.std(data.iloc[:, :2])  # 归一化
    x = cap.feature_mapping(data['score1'], data['score2'], degree=8)                             # 决策变量
    y = np.array([data['result']]).T                                                              # 标签

    model = logistic(x, y)                           # 实例化一个Logistic类
    model.train(epochs=10000, alpha=0.05, lamda=20)  # 训练
    pred = model.predict(x)                          # 预测
    model.plot_original_data(data)                   # 可视化原始样本分布
    model.plot_decision_boundary(data)               # 可视化决策边界



