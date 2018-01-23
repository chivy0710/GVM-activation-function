#!/usr/bin/env python
# -*- coding:utf-8 -*-

"Clement Xu Program"
"本程序目的是读取相关数据"

import csv
import numpy as np
import matplotlib.pyplot as plt

def ReadElectricity():
    csv_file = csv.reader(open('./Data_Electricity/data_monday.csv', 'r'))
    rows = [row for row in csv_file]
    data = []
    len_row = len(rows)
    len_col = len(rows[0])
    for i in range(len_row):
        data = data + [[float(j) for j in rows[i]]]
    data = np.array(data)
    maxE = np.max(data)
    minE = np.min(data)
    # 归一化处理，使数据映射到0到1之间
    data = (data - minE) / (maxE - minE)

    # print(data)

    training_in = data[0:48, 0:7]
    training_out = data[0:48, 7]
    test_in = data[0:48, 1:8]
    test_out = data[0:48, 8]
    return training_in, training_out, test_in, test_out, maxE, minE

def ReadCancerData():
    # 数据格式为输入9个参数，输出标签为2分类问题，即是否有癌症，1为有，-1为没有
    # file1 为训练的输入数据,file2 为训练标签, file3 为测试的输入数据,file4 为测试标签
    file1 = './Data_Cancer/breast-cancer_train_data_1.asc'
    file2 = './Data_Cancer/breast-cancer_train_labels_1.asc'
    file3 = './Data_Cancer/breast-cancer_test_data_1.asc'
    file4 = './Data_Cancer/breast-cancer_test_labels_1.asc'

    data1 = np.loadtxt(file1, dtype=float)
    data2 = np.loadtxt(file2, dtype=float)
    data3 = np.loadtxt(file3, dtype=float)
    data4 = np.loadtxt(file4, dtype=float)

    np.reshape(data2, len(data2))
    np.reshape(data4, len(data4))
    data2[data2 == -1.0] = 0
    data4[data4 == -1.0] = 0

    training_in = data1
    training_out = data2

    test_in = data3
    test_out = data4

    return training_in, training_out, test_in, test_out


if __name__ == '__main__':

    # training_in, training_out, test_in, test_out = ReadMulFunction()
    # print("training_in:", training_in)
    # print("training_out:", training_out)
    # print("test_in:", test_in)
    # print("test_out:", test_out)
    # for i in range(5):
    #     MulFunctionPlot(training_in,training_out,i)

    # training_in, training_out, test_in, test_out, maxE, minE = ReadElectricity()
    # # print(training_in)
    # # print(training_out)
    # # print(test_in,test_out)
    #
    # x = np.linspace(1, 48, 48)
    # plt.figure(1)
    # for i in range(7):
    #     plt.plot(x, training_in[:,i])
    # plt.show()