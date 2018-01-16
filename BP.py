#! /usr/bin/env python
# -*- coding: utf-8 -*-

"利用pytorch的构架来实现BP神经网络"
"本示例用于实现澳大利亚电力数据的拟合实验,每天半个小时采样一点,采样为48个点,输入数据为7周相同时刻的点,输出为第8周同时刻的点."
"模型结构为，输入-隐藏-输出：7-50-1"

import torch
from torch.autograd import Variable
import matplotlib.pyplot as plt
import numpy as np
import math
import Read_Data


# 读取数据,由于做归一化处理，保留最大最小值
training_in, training_out, test_in, test_out, maxE, minE = Read_Data.ReadElectricity()
# 将numpy的array格式转换为torch的Tensor，记得将类型转为floatTensor
x = torch.from_numpy(training_in).float()
y = torch.from_numpy(training_out).float()

# 转换为Variable的类型
x = Variable(x)
y = Variable(y)


x_test = torch.from_numpy(test_in).float()
y_test = torch.from_numpy(test_out).float()
x_test = Variable(x_test)
y_test = Variable(y_test)

step = 7000

D_in, H, D_out = 7, 50, 1

# 定义模型，线性，sigmoid函数作为激活函数
model = torch.nn.Sequential(
    torch.nn.Linear(D_in, H),
    torch.nn.Tanh(),
    torch.nn.Linear(H, D_out),
)

# print(model.parameters())

# 定义MSE为loss function
loss_fn = torch.nn.MSELoss(size_average=False)

learning_rate = 1e-4
# 定义梯度下降优化选择为adagrad
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
# k 定义的是loss的值
k = 10000
t = 0
while t < step and k> 0.02:
    t += 1
    y_pred = model(x)
    loss = loss_fn(y_pred, y)
    k = loss.data[0]
    if t % 100 == 0:
        print(t, k)
    # 梯度清零，避免重复计算
    optimizer.zero_grad()
    # 迭代
    loss.backward()
    optimizer.step()
x = x.data.numpy()
y = y.data.numpy()
y_pred = y_pred.data.numpy()
# x = x.reshape

y_test_pred = model(x_test)
loss_test = loss_fn(y_test_pred, y_test)

x_test = x_test.data.numpy()
y_test = y_test.data.numpy()
y_test_pred = y_test_pred.data.numpy()

print("Loss is:", loss_test.data[0])

# 画图部分
x_axis = np.linspace(1,48,48)
plt.figure(1)
plt.plot(x_axis, y_test, color='steelblue')
plt.plot(x_axis, y_test_pred, color='tomato')
plt.show()
