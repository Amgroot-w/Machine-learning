"""
《机器学习》作业 --- 1.线性回归

学号：2000900
姓名：张敬川
11.30

调试记录：

"""
# %% 导入包
import numpy as np
import tensorflow as tf
import pandas as pd
import matplotlib.pyplot as plt
# 显示中文
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# %% 定义LR类
class LR(object):

    def __init__(self, x, y):
        self.x = x  # 决策变量
        self.y = y  # 函数值

    # 训练
    def train(self, alpha=0.01, epochs=1000):
        m = self.x.shape[0]    # 样本个数
        n = self.x.shape[1]    # 变量个数
        self.alpha = alpha     # 学习率
        self.epochs = epochs   # 迭代次数
        # 绘图
        self.x = tf.placeholder(tf.float32, [None, n])
        self.y = tf.placeholder(tf.float32, [None, 1])
        theta = tf.Variable(tf.random_normal([n, 1]))
        pred = tf.matmul(self.x, theta)
        J = 1/2 * tf.reduce_mean(tf.pow(pred-self.y, 2))
        theta_update = 1/m * self.alpha * tf.matmul(tf.transpose(self.x), pred-self.y)
        theta = tf.assign_add(theta, -theta_update)

        # 启动会话
        with tf.Session() as sess:
            self.cost_history = {'epoch': [], 'cost': []}
            sess.run(tf.global_variables_initializer())

            print('********** 开始训练 **********')
            for epoch in range(epochs):
                self.thetas, cost = sess.run([theta, J], feed_dict={self.x: train_x, self.y: train_y})  # 喂数据
                self.cost_history['epoch'].append(epoch)  # 保存每次迭代的cost数据
                self.cost_history['cost'].append(cost)
                print('epoch:%3d   cost:%4f' % (epoch, cost))
            print('********** 训练完成 **********')

        print(self.thetas)

    # 预测
    def predict(self, x):
        return self.thetas[0] + x*self.thetas[1]

    # 可视化
    def show(self, data):
        plt.plot(data.iloc[:, 0], data.iloc[:, 1], '.')  # 原始样本
        plt.plot([5, 25], [self.predict(5), self.predict(25)], '--k')  # 绘制分界线
        plt.xlabel('x')
        plt.ylabel('y')
        plt.title('线性回归')
        plt.show()


# %% 数据集测试
if __name__ == '__main__':
    data = pd.read_csv('data_LR.csv', header=None)   # 读取cvs文件
    plt.plot(data.iloc[:, 0], data.iloc[:, 1], '.')  # 绘制原始样本
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('原始样本分布')
    plt.show()
    train_x = pd.DataFrame(np.column_stack((np.ones(data.shape[0]), data.iloc[:, 0].values)))  # 加入第一列（全1）
    train_y = pd.DataFrame(data.iloc[:, 1])  # 函数值

    model = LR(train_x, train_y)                # 实例化一个类
    model.train(alpha=0.01, epochs=1000)        # 训练
    pred = model.predict(train_x.values[:, 1])  # 测试
    model.show(data)                            # 可视化







