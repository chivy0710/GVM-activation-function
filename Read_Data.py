#!/usr/bin/env python
# -*- coding:utf-8 -*-

"Clement Xu Program"
"本程序目的是读取相关数据"

import csv
import numpy as np
import matplotlib.pyplot as plt

def ReadMulFunction():
    csv_file = csv.reader(open('./Data_Generation/MulFuncData.csv', 'r'))
    rows = [row for row in csv_file]
    data = []
    len_row = len(rows)
    len_col = len(rows[0])
    for i in range(len_row):
        data = data + [[float(j) for j in rows[i]]]
    data = np.array(data)

    training_in = data[[0]]
    training_out = data[[20]]
    test_in = data[[40]]
    test_out = data[[41]]
    return training_in, training_out, test_in, test_out

def MulFunctionPlot(x,y,i):
    plt.figure(i)
    plt.plot(x[i],y[i])
    plt.show()

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

def ReadZooData():
    csv_file = csv.reader(open('./Data_zoo/zoo.csv', 'r'))
    rows = [row for row in csv_file]
    data = []
    len_row = len(rows)
    len_col = len(rows[0])
    for i in range(len_row):
        data = data + [[float(j) for j in rows[i]]]
    data = np.array(data)
    # print(data)
    # print(len_row,len_col)

    class_zoo = {1:[1,0,0,0,0,0,0], 2 :[0,1,0,0,0,0,0], 3:[0,0,1,0,0,0,0], 4:[0,0,0,1,0,0,0],
                 5: [0, 0, 0, 0, 1, 0, 0], 6:[0,0,0,0,0,1,0], 7:[0,0,0,0,0,0,1]}
    training_in = data[0:81, 0:16]
    training_out_value = data[0:81, 16]
    training_out = []
    for i in training_out_value:
        training_out.append(class_zoo[i])
    training_out = np.array(training_out)
    # print(training_out)
    test_in = data[81:101, 0:16]
    test_out_value = data[81:101, 16]
    test_out = []
    for i in test_out_value:
        test_out.append(class_zoo[i])
    test_out = np.array(test_out)
    # print(test_out)
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
    training_in, training_out, test_in, test_out = ReadZooData()