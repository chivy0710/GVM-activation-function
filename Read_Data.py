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

    train_data = np.array([])
    file = []
    for i in range(100):
        file.append("./Data_Cancer/breast-cancer_train_data_"+ str(i+1) +".asc")
        file_load = np.loadtxt(file[i], dtype=float)
        train_data = np.append(train_data, file_load)
    train_data = np.reshape(train_data, [20000, 9])

    train_labels = np.array([])
    file = []
    for i in range(100):
        file.append("./Data_Cancer/breast-cancer_train_labels_"+ str(i+1) +".asc")
        file_load = np.loadtxt(file[i], dtype=float)
        train_labels = np.append(train_labels, file_load)
    # train_labels = np.reshape(train_labels, [20000, 1])

    test_data = np.array([])
    file = []
    for i in range(100):
        file.append("./Data_Cancer/breast-cancer_test_data_" + str(i + 1) + ".asc")
        file_load = np.loadtxt(file[i], dtype=float)
        test_data = np.append(test_data, file_load)
    test_data = np.reshape(test_data, [7700, 9])

    test_labels = np.array([])
    file = []
    for i in range(100):
        file.append("./Data_Cancer/breast-cancer_test_labels_" + str(i + 1) + ".asc")
        file_load = np.loadtxt(file[i], dtype=float)
        test_labels = np.append(test_labels, file_load)

    train_labels[train_labels == -1.0] = 0
    test_labels[test_labels == -1.0] = 0

    train_size, test_size = 5000, 50


    training_in = train_data[0:train_size,]
    training_out = train_labels[0:train_size,]

    test_in = test_data[0:test_size,]
    test_out = test_labels[0:test_size,]

    return training_in, training_out, test_in, test_out

def ReadCancerDataClass():
    # 数据格式为输入9个参数，输出标签为2分类问题，即是否有癌症，1为有，-1为没有

    train_data = np.array([])
    file = []
    for i in range(100):
        file.append("./Data_Cancer/breast-cancer_train_data_"+ str(i+1) +".asc")
        file_load = np.loadtxt(file[i], dtype=float)
        train_data = np.append(train_data, file_load)
    train_data = np.reshape(train_data, [20000, 9])

    train_labels = np.array([])
    train_labels_bi = []
    file = []
    for i in range(100):
        file.append("./Data_Cancer/breast-cancer_train_labels_"+ str(i+1) +".asc")
        file_load = np.loadtxt(file[i], dtype=float)
        train_labels = np.append(train_labels, file_load)
    train_labels[train_labels == -1.0] = 0

    for i in range(len(train_labels)):
        if train_labels[i] == 0:
            train_labels_bi.append([0, 1])
        else:
            train_labels_bi.append([1, 0])
    train_labels_bi = np.array(train_labels_bi)

    test_data = np.array([])
    file = []
    for i in range(100):
        file.append("./Data_Cancer/breast-cancer_test_data_" + str(i + 1) + ".asc")
        file_load = np.loadtxt(file[i], dtype=float)
        test_data = np.append(test_data, file_load)
    test_data = np.reshape(test_data, [7700, 9])

    test_labels = np.array([])
    test_labels_bi = []
    file = []
    for i in range(100):
        file.append("./Data_Cancer/breast-cancer_test_labels_" + str(i + 1) + ".asc")
        file_load = np.loadtxt(file[i], dtype=float)
        test_labels = np.append(test_labels, file_load)
    test_labels[test_labels == -1.0] = 0
    for i in range(len(test_labels)):
        if test_labels[i] == 0:
            test_labels_bi.append([0, 1])
        else:
            test_labels_bi.append([1, 0])
    test_labels_bi = np.array(test_labels_bi)

    train_size, test_size = 5000, 50


    training_in = train_data[0:train_size,]
    training_out = train_labels_bi[0:train_size,]

    test_in = test_data[0:test_size,]
    test_out = test_labels_bi[0:test_size,]
    print(test_out)

    return training_in, training_out, test_in, test_out

if __name__ == '__main__':
    training_in, training_out, test_in, test_out = ReadCancerDataClass()
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