#!/usr/bin/env python
# -*- coding:utf-8 -*-

"Clement Xu Program"
"本程序是GVM模型的主程序，用来测试不同激活函数对函数拟合，电力预测的效果"

import numpy as np
import math,copy
import random
import matplotlib.pyplot as plt
import os
import Read_Data


class GVM:
    # 定义激活函数，包含sigmoid函数，tanh函数，gauss函数，softplus函数，ReLu函数
    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def gauss(self, x):
        return np.exp(-np.square(x))

    def tanh(self, x):
        return np.tanh(x)

    def softplus(self, x):
        return np.log(1 + np.exp(x))

    def Relu(self, x):
        return np.maximum(x, 0)

# Important parameters

    def __init__(self, train_x=np.array([]), train_y=np.array([]), test_x = np.array([]), test_y = np.array([]),
                 M=5,N=10,L=2,MC_step=10000,
                 Endloss=5,c_beta=0.8,c_weight1=1.0,c_bias=1.0, Numfunc=[]):

        #initialize input layer M, hidden layer N, output layer L
        self.N = N
        self.M = M
        self.L = L

        self.transfer_dict = {
            0:self.sigmoid,
            1:self.gauss,
            2:self.tanh,
            3:self.softplus,
            4:self.Relu
        }
        self.Numfunc = Numfunc
        self.NumfuncRange = [Numfunc[0], Numfunc[0] + Numfunc[1], Numfunc[0] + Numfunc[1] + Numfunc[2],
                             Numfunc[0] + Numfunc[1] + Numfunc[2] + Numfunc[3],
                             Numfunc[0] + Numfunc[1] + Numfunc[2] + Numfunc[3] + Numfunc[4]]
        self.train_x = train_x
        self.train_y = train_y
        self.test_x = test_x
        self.test_y = test_y

        self.dataNum = self.train_x.shape[0]
        self.dataNumTest = self.test_x.shape[0]
        self.c_beta = c_beta
        self.c_weight1 = c_weight1
        self.c_bias = c_bias

        self.weight1 = 2 * self.c_weight1 * np.random.random((self.M, self.N)) - self.c_weight1
        self.bias = 2 * self.c_bias * np.random.random((self.N, 1)) - self.c_bias
        self.beta = 2 * self.c_beta * np.random.random((self.N, 1)) - self.c_beta

        # range of beta
        self.weight2 = np.random.randint(0, 2, size = (self.N, self.L))
        self.weight2[self.weight2 == 0] = -1
        self.MC_step = MC_step
        self.Endloss = Endloss
        self.tempError = 10000000

    def ElectricityPlot(self, y_real, y_pred):
        x = np.linspace(1, 48,48)
        plt.plot(x,y_real,color='steelblue')
        plt.plot(x,y_pred,color='tomato')
        plt.show()

    def loss_fn(self, x, y):
        shape_x = x.shape
        shape_y = y.shape
        if shape_x == shape_y:
            loss = np.sum((x - y) ** 2)
        else:
            print("Error of size!")
        return loss

    def mc_random(self, weight, c_weight):
        while True:
            change = 0.1 * ((2 * random.random() - 1))
            if -c_weight < weight + change < c_weight:
                break
        return change

    def train(self):
        local = np.zeros([self.dataNum, self.N])
        hidden = np.zeros([self.dataNum,self.N])
        if self.L == 1:
            output = np.zeros(self.dataNum)
        else:
            output = np.zeros([self.dataNum, self.L])

        for i in range(self.dataNum):
            for n in range(self.N):
                for m in range(self.M):
                    local[i,n] += self.weight1[m,n] * self.train_x[i,m]
            for n in range(self.N):
                local[i,n] += self.bias[n][0]
                for k in range(5):
                    if n < self.NumfuncRange[k]:
                        hidden[i,n] = self.transfer_dict[k](local[i,n] * self.beta[n][0])
                        break
                for l in range(self.L):
                    if self.L == 1:
                        output[i] +=self.weight2[n,l] * hidden[i,n]
                    else:
                        output[i,l] += self.weight2[n,l] * hidden[i,n]

        loss = self.loss_fn(output, self.train_y)

        avg_loss = loss

        epoch = 0

        while epoch < self.MC_step and avg_loss > self.Endloss:
            epoch +=1
            if (epoch % 1000) == 0 or (epoch == 1):
                print("Epoch:", epoch, "The loss is:", avg_loss)

            change_class = np.random.randint(0, 3)

            # change_class = 0 对应于更改weight1的一个权值
            if change_class == 0:
                # MC1 对应隐藏层, MC2 对应输入层
                MC1 = np.random.randint(0, self.N)
                MC2 = np.random.randint(0, self.M)
                change = self.mc_random(self.weight1[MC2,MC1], self.c_weight1)
                # 深度拷贝local, hidden, output
                new_local = copy.deepcopy(local)
                new_hidden = copy.deepcopy(hidden)
                new_output = copy.deepcopy(output)

                for i in range(self.dataNum):
                    new_local[i,MC1] += change * self.train_x[i,MC2]
                    for k in range(5):
                        if MC1 < self.NumfuncRange[k]:
                            new_hidden[i,MC1] = self.transfer_dict[k](new_local[i,MC1] * self.beta[MC1][0])
                            break
                    for l in range(self.L):
                        if self.L == 1:
                            new_output[i] -= self.weight2[MC1,l] * hidden[i,MC1]
                            new_output[i] += self.weight2[MC1,l] * new_hidden[i,MC1]
                        else:
                            new_output[i, l] -= self.weight2[MC1, l] * hidden[i, MC1]
                            new_output[i, l] += self.weight2[MC1, l] * new_hidden[i, MC1]

                tmp_loss = self.loss_fn(new_output, self.train_y)
                tmp_avg_loss = tmp_loss
                if (avg_loss - tmp_avg_loss) >= 0:
                    self.weight1[MC2,MC1] += change
                    avg_loss = tmp_avg_loss
                    local = copy.deepcopy(new_local)
                    hidden = copy.deepcopy(new_hidden)
                    output = copy.deepcopy(new_output)


            # change_class = 1 对应于更改bias的一个权值
            if change_class == 1:
                # MC1 对应隐藏层
                MC1 = np.random.randint(0, self.N)
                change = self.mc_random(self.bias[MC1][0], self.c_bias)
                # 深度拷贝local, hidden, output
                new_local = copy.deepcopy(local)
                new_hidden = copy.deepcopy(hidden)
                new_output = copy.deepcopy(output)

                for i in range(self.dataNum):
                    new_local[i,MC1] += change
                    for k in range(5):
                        if MC1 < self.NumfuncRange[k]:
                            new_hidden[i,MC1] = self.transfer_dict[k](new_local[i,MC1] * self.beta[MC1][0])
                            break
                    for l in range(self.L):
                        if self.L == 1:
                            new_output[i] -= self.weight2[MC1,l] * hidden[i,MC1]
                            new_output[i] += self.weight2[MC1,l] * new_hidden[i,MC1]
                        else:
                            new_output[i, l] -= self.weight2[MC1, l] * hidden[i, MC1]
                            new_output[i, l] += self.weight2[MC1, l] * new_hidden[i, MC1]

                tmp_loss = self.loss_fn(new_output, self.train_y)
                tmp_avg_loss = tmp_loss
                if (avg_loss - tmp_avg_loss) >= 0:
                    self.bias[MC1][0] += change
                    avg_loss = tmp_avg_loss
                    local = copy.deepcopy(new_local)
                    hidden = copy.deepcopy(new_hidden)
                    output = copy.deepcopy(new_output)

            # change_class = 2 对应于更改beta的一个权值
            if change_class == 2:
                # MC1 对应隐藏层
                MC1 = np.random.randint(0, self.N)
                change = self.mc_random(self.beta[MC1][0], self.c_beta)
                # 深度拷贝hidden, output
                new_hidden = copy.deepcopy(hidden)
                new_output = copy.deepcopy(output)

                for i in range(self.dataNum):
                    for k in range(5):
                        if MC1 < self.NumfuncRange[k]:
                            new_hidden[i, MC1] = self.transfer_dict[k](local[i, MC1] * (self.beta[MC1][0] + change))
                            break
                    for l in range(self.L):
                        if self.L == 1:
                            new_output[i] -= self.weight2[MC1,l] * hidden[i,MC1]
                            new_output[i] += self.weight2[MC1,l] * new_hidden[i,MC1]
                        else:
                            new_output[i, l] -= self.weight2[MC1, l] * hidden[i, MC1]
                            new_output[i, l] += self.weight2[MC1, l] * new_hidden[i, MC1]

                tmp_loss = self.loss_fn(new_output, self.train_y)
                tmp_avg_loss = tmp_loss
                if (avg_loss - tmp_avg_loss) >= 0:
                    self.beta[MC1][0] += change
                    avg_loss = tmp_avg_loss
                    hidden = copy.deepcopy(new_hidden)
                    output = copy.deepcopy(new_output)

        # self.ElectricityPlot(self.train_y, output)
        return epoch

    def test(self):
        local = np.zeros([self.dataNum, self.N])
        hidden = np.zeros([self.dataNum, self.N])
        if self.L == 1:
            output = np.zeros(self.dataNum)
        else:
            output = np.zeros([self.dataNum, self.L])

        for i in range(self.dataNumTest):
            for n in range(self.N):
                for m in range(self.M):
                    local[i, n] += self.weight1[m, n] * self.test_x[i, m]
            for n in range(self.N):
                local[i, n] += self.bias[n][0]
                for k in range(5):
                    if n < self.NumfuncRange[k]:
                        hidden[i, n] = self.transfer_dict[k](local[i, n] * self.beta[n][0])
                        break
                for l in range(self.L):
                    if self.L == 1:
                        output[i] += self.weight2[n, l] * hidden[i, n]
                    else:
                        output[i, l] += self.weight2[n, l] * hidden[i, n]

        loss = self.loss_fn(output, self.test_y)
        avg_loss = loss

        # self.ElectricityPlot(self.test_y, output)

        return avg_loss

    def save_gvm(self):
        np.savez(os.path.join(os.getcwd(), 'GVMinit.npz'), weight1=self.weight1, bias=self.bias,
                 beta=self.beta, weight2=self.weight2)

    def load_gvm(self):
        r = np.load(os.path.join(os.getcwd(), 'GVMinit.npz'))
        self.weight1 = r['weight1']
        self.bias = r['bias']
        self.beta = r['beta']
        self.weight2 = r['weight2']

def GenNumfunc(N):
    Numfunc = []
    sum = N
    for i in range(4):
        a = random.randint(0,sum)
        sum -= a
        Numfunc.append(a)
    Numfunc.append(sum)
    return  Numfunc


if __name__ == '__main__':

    # 读取电力数据，输入数据为48组，7条作为输入，1条作为输出
    training_in_e, training_out_e, test_in_e, test_out_e, maxE, minE = Read_Data.ReadElectricity()

    # 定义一些必要参数,N为隐藏层节点数,Numfunc[5]为5种不同激活函数对应神经节点的数量,和为N
    # 对应的为sigmoid, gauss, tanh, softplus, ReLu

    M_e, N_e, L_e = 7, 50, 1
    # 截止的迭代次数以及最终误差
    step_e = 200000
    Endloss_e = 0.05

    # Numfunc_e = [0, 0, 0, 0, 50]
    # 利用随机性来寻找最好激活函数构成组合

    best_epoch = step_e
    best_loss = 10000
    best_iter = [0,0]
    Num_iter = 50
    Numfunc_e = np.zeros([Num_iter,5])
    GVM_Loss = np.zeros([Num_iter, 7])
    epoch_training = np.zeros([Num_iter, 7])
    for time in range(Num_iter):
        Numfunc_e[time] = GenNumfunc(N_e)
        print("The iteration is:",time,"\n**********************")
        for i in range(7):
            gvm_e = GVM(train_x=training_in_e, train_y=training_out_e, test_x=test_in_e, test_y=test_out_e,
                        M=M_e, N=N_e, L=L_e,
                       MC_step=step_e, Endloss=Endloss_e, Numfunc=Numfunc_e[time])
            gvm_e.load_gvm()
            epoch_training[time,i] = gvm_e.train()
            GVM_Loss[time,i] = gvm_e.test()
        avg_epoch = np.average(epoch_training[time])
        avg_loss = np.average(GVM_Loss[time])
        if avg_loss < best_loss:
            best_iter[0] = time
        if avg_epoch < best_epoch:
            best_iter[1] = time

    print("Training epoch:\n", epoch_training)
    print("The loss:\n", GVM_Loss)

    print("The best result of epoch:\n",Numfunc_e[best_iter[0]])
    print("The avearge of epoch:", np.average(epoch_training[best_iter[0]]),
          "details:\n", epoch_training[best_iter[0]])
    print("The avearge of loss:", np.average(GVM_Loss[best_iter[0]]),
          "details:\n", GVM_Loss[best_iter[0]])

    print("The best result of loss:\n", Numfunc_e[best_iter[1]])
    print("The avearge of epoch:", np.average(epoch_training[best_iter[1]]),
          "details:\n", epoch_training[best_iter[1]])
    print("The avearge of loss:", np.average(GVM_Loss[best_iter[1]]),
          "details:\n", GVM_Loss[best_iter[1]])
