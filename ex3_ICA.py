"""
《机器学习》作业 --- 3. 独立成分分析（ICA）

学号：2000900
姓名：张敬川
12.10

调试记录：

"""
import math
import numpy as np
import random
import matplotlib.pyplot as plt
from numpy import *

# 显示中文
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

def f1(x, period=4):
    return 0.5*(x-math.floor(x/period)*period)

def create_data():
    # 数据量
    n = 500
    # 记录时间
    T = [0.1*xi for xi in range(0, n)]
    # 声源信号
    S = array([[sin(xi) for xi in T], [f1(xi) for xi in T]], float32)
    # 混合矩阵
    A = np.random.uniform(0, 1, [2, 2])
    return T, S, dot(A, S)

def whiten(X):
    # 中心化
    X_mean = X.mean(axis=-1)
    X -= X_mean[:, newaxis]
    # 白化
    A = dot(X, X.transpose())
    D, E = linalg.eig(A)
    D2 = linalg.inv(array([[D[0], 0.0], [0.0, D[1]]], float32))
    D2[0, 0] = sqrt(D2[0, 0])
    D2[1, 1] = sqrt(D2[1, 1])
    V = dot(D2, E.transpose())
    return dot(V, X), V

def _logcosh(x, fun_args=None, alpha=1):
    gx = tanh(alpha * x, x)
    g_x = gx ** 2
    g_x -= 1.
    g_x *= -alpha
    return gx, g_x.mean(axis=-1)

def do_decorrelation(W):
    s, u = linalg.eigh(dot(W, W.T))
    return dot(dot(u * (1. / sqrt(s)), u.T), W)

# ICA算法
def do_fastica(X):
    n, m = X.shape
    p = float(m)
    g = _logcosh
    X *= sqrt(X.shape[1])
    # 初始化w
    W = ones((n, n), float32)
    for i in range(n):
        for j in range(i):
            W[i, j] = random.random()
    # 计算w
    maxIter = 200
    for ii in range(maxIter):
        gwtx, g_wtx = g(dot(W, X))
        W1 = do_decorrelation(dot(gwtx, X.T) / p - g_wtx[:, newaxis] * W)
        lim = max(abs(abs(diag(dot(W1, W.T))) - 1))
        W = W1
        if lim < 0.0001:
            break
    return W

def show_data(T, S, name):
    plt.subplot(211)
    plt.plot(T, [S[0, i] for i in range(S.shape[1])])
    plt.title(name + '1')
    plt.subplot(212)
    plt.plot(T, [S[1, i] for i in range(S.shape[1])])
    plt.title(name + '2')
    plt.show()


if __name__ == "__main__":
    T, S, D = create_data()  # 产生数据
    Dwhiten, K = whiten(D)  # 白化
    W = do_fastica(Dwhiten)  # 迭代计算W
    Sr = dot(dot(W, K), D)  # 复原的信号

    show_data(T, S, '声源信号')
    show_data(T, D, '混合信号')
    show_data(T, Sr, '复原信号')

