#!/usr/bin/env python
# -*- coding:utf-8 -*-

"Clement Xu Program"
"本程序目的是生成连续分段函数，包含反三角函数，三角函数，对数函数，指数函数，sigmoid函数，gauss函数以及多项式函数。"
"构造的连续函数加上随机噪声作为数据，生成70个点的11条数据，并打算以其中10条为训练集，1条作为测试集。"

import matplotlib.pyplot as plt
import numpy as np
import math
import csv

# y = asin(x - 1), x = [0, 1]
x = np.array([])
y = np.array([])
x1 = np.linspace(-1, 0, 41)
y1 = [math.asin(i) for i in x1]
x = np.append(x, np.linspace(0, 1, 41))
y = np.append(y, y1)

# y = sin(x - 1), x =[1, 2Pi + 1]
x2 = np.linspace(-3.14, 3.14, 91)
y2 = [math.sin(i) for i in x2]
x2 = x2 + 4.14
x = np.append(x, x2)
y = np.append(y, y2)

# y = ln(x - 2Pi), x =[2Pi+1, 2Pi+1+e]
x3 = np.linspace(1, 2.718, 30)
y3 = [math.log(i) for i in x3]
x3 = x3 + 2 * 3.14
x = np.append(x, x3)
y = np.append(y, y3)

# y = -sigmoid(5(x-2Pi-3-e)), x = [2Pi+1+e, 2Pi+5+e]
x4 = np.linspace(-2, 2, 61)
y4 = [- (1 /( 1 + math.exp(- 5 * i))) + 1 for i in x4]
x4 = x4 + 2 * 3.14 + 2.718 + 2
x = np.append(x, x4)
y = np.append(y, y4)

# y = gauss(x -2Pi-e-8), x = [2Pi+e+5, 2Pi+e+11]
x5 = np.linspace(-3, 3, 81)
y5 = [math.exp(- i * i) for i in x5]
x5 = x5 + 2 * 3.14 + 2.718 + 7
x = np.append(x, x5)
y = np.append(y, y5)

# y = exp(x-(2Pi+e+17)), x =[2Pi+e+11,2pi+e+18]
x6 = np.linspace(-7, 0, 101)
y6 = [math.exp(i) for i in x6]
x6 = x6 + 2 * 3.14 + 2.718 + 17
x = np.append(x, x6)
y = np.append(y, y6)

#
x7 = np.linspace(-1, 2, 95)
y7 =[ -i*(i+1)*(i-1.8)*(i-1.2)*(i-0.9)+1 for i in x7]
x7 = x7 + 2 * 3.14 + 2.718 + 18
x = np.append(x, x7)
y = np.append(y, y7)

# add noise
length = np.size(x)

# 取范围内的100个均匀分布的点，且不加噪声作为测试集
test_in = x[0:500:10]
test_out = y[0:500:10]

print(length)
noise = 0.1 * (2 * np.random.random(length) - 1)
y_noise = y + noise

plt.figure()
plt.plot(x, y)
# plt.scatter(x, y_noise)
plt.show()

# data generation
data = np.array([])
sizeC = 50
for i in range(20):
    a = np.random.randint(0, 500, size=sizeC)
    a.sort()
    print(a)
    data = np.append(data, a)
# data = data.reshape((11, sizeC))
print(data)
x_write = np.array([])
y_write = np.array([])
for i in data:
    x_write = np.append(x_write, x[int(i)])
    y_write = np.append(y_write, y_noise[int(i)])

x_write = x_write.reshape((20, sizeC))
y_write = y_write.reshape((20, sizeC))
print(x_write[0])
print(y_write[0])

plt.figure()
for i in range(20):
    plt.plot(x_write[i], y_write[i])
plt.plot(x,y)
# plt.scatter(x, y_noise)
plt.show()

# 读写文件
csvFile = open("MulFuncData.csv", "w")
writer = csv.writer(csvFile)
writer.writerows([i for i in x_write])
writer.writerows([j for j in y_write])
writer.writerow([i for i in test_in])
writer.writerow([j for j in test_out])
csvFile.close()
