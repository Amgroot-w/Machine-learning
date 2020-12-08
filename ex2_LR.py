# 线性回归--多变量
# Feb.16th
# 调试记录：
# 1. 如果不归一化，模型为什么会发散的，无法收敛？？？？？
#       其实不是，单变量不需要归一化模型也可以收敛，而之前发散的原因是：把公式写错了：θ的更新
#    少乘了1/m（主要原因），J(θ)少乘了1/2.
# 2. 训练模型得到了适用于归一化后数据的参数theta，但是给定一个原始scale的x，怎么得到预测值？
#       给定测试值，将其按照训练集的数据（均值、方差）归一化，然后带入模型得到预测值。
# 3. 归一化是只对X，不对Y？
#       归一化的目的是解决各个特征的量度不统一问题，有的特征值整体很大，有的整体很小，这很容
#    易造成某一个特征对模型的训练器决定作用而忽视了其他特征。因此要对特征进行归一化，而不是Y
#    值（这样也能保证将测试值带入模型后，得到的预测结果就是实际的Y值量度）。

import numpy as np
import tensorflow as tf
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# 导入数据
data0 = pd.read_csv('data_LR.csv', header=None)  # 读取cvs文件
data0_x = data0.iloc[:, :2]
data0_y = data0.iloc[:, 2]
data_x = (data0_x - np.mean(data0_x)) / np.std(data0_x)  # 对特征x归一化
data_y = data0_y  # y值不进行归一化
data = pd.DataFrame(np.column_stack((data_x, data_y)))

train_x = pd.DataFrame(np.column_stack((np.ones(data.shape[0]),
                                        data.iloc[:, :2].values)))  # 加入第一列（全1）
train_y = pd.DataFrame(data.iloc[:, 2])

# 搭建TensorFlow模型
# 参数
m = train_x.shape[0]  # 样本个数
n = train_x.shape[1]  # 变量个数
learning_rate = 0.01  # 学习率
epochs = 1000  # 迭代次数
# 绘图
x = tf.placeholder(tf.float32, [None, n])
y = tf.placeholder(tf.float32, [None, 1])
theta = tf.Variable(tf.random_normal([n, 1]))
pred = tf.matmul(x, theta)
J = 1/2 * tf.reduce_mean(tf.pow(pred-y, 2))
theta_update = 1/m * learning_rate * tf.matmul(tf.transpose(x), pred-y)
theta = tf.assign_add(theta, -theta_update)

# 启动会话
with tf.Session() as sess:
    cost_history = {'epoch': [], 'cost': []}
    sess.run(tf.global_variables_initializer())

    print('********** 开始训练 **********')
    for epoch in range(epochs):
        t, cost = sess.run([theta, J], feed_dict={x: train_x, y: train_y})  # 喂数据
        cost_history['epoch'].append(epoch)  # 保存每次迭代的cost数据
        cost_history['cost'].append(cost)
        print('epoch:%3d   cost:%4f' % (epoch, cost))
    print('********** 训练完成 **********')

# 可视化
plt.plot(cost_history['epoch'], cost_history['cost'])  # cost曲线
plt.show()

test_x0 = pd.DataFrame([[800, 1],  # 生成两个测试样本
                        [1650, 3],
                        [4500, 4]])
test_x = (test_x0 - np.mean(data0_x)) / np.std(data0_x)  # 归一化
test_x = pd.DataFrame(np.column_stack((np.ones(3), test_x.values)))  # 补第一列
test_pred = np.matmul(test_x, t)  # 输入模型，得到预测值
print('1650-square-foot house with 3 bedrooms, its price is about: %.2f$' % test_pred[1])

ax = Axes3D(plt.figure())
ax.plot(data0.iloc[:, 0], data0.iloc[:, 1], data0.iloc[:, 2], '.')  # 原始样本分布
ax.plot(test_x0.iloc[:, 0], test_x.iloc[:, 1], test_pred[:, 0])  # 拟合曲线
plt.show()






