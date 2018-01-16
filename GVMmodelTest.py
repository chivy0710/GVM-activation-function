import numpy as np
import math

class GVM:
    # 定义激活函数，包含sigmoid函数，tanh函数，gauss函数，softplus函数，ReLu函数
    def sigmoid(self, x):
        return x + 1

    def gauss(self, x):
        return x + 2

    def tanh(self, x):
        return x + 3

    def softplus(self, x):
        return x + 4

    def Relu(self, x):
        return x + 5

# Important parameters

    def __init__(self, train_x=np.array([]), train_y=([]),M=3,N=5,L=3,Numfunc=[1,1,1,1,1]):

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
        self.dataNum = self.train_x.shape[0]
        # print(self.dataNum)

        self.weight1 = [[1,2,3,4,5],[2,3,4,5,6],[3,4,5,6,7]]
        self.bias = [[1],[2],[3],[4],[5]]
        self.beta = [[0.1],[0.2],[0.3],[0.4],[0.5]]

        # range of beta
        self.weight2 = [[-1,1,1],[1,1,1],[-1,-1,-1],[-1,1,-1],[1,-1,1]]

    # 定义每次参数迭代增加的小量，引入avg_loss项目的是使得在训练接近尾声的时候步幅减小

    # training
    def train(self):
        local = np.zeros([self.dataNum,self.N])
        output = np.zeros([self.dataNum,self.L])
        for i in range(self.dataNum):
            for n in range(self.N):
                for m in range(self.M):
                    local[i][n] +=self.train_x[i][m] * self.weight1[m][n]
                local[i][n] +=self.bias[n][0]
                for k in range(5):
                    if n < self.NumfuncRange[k]:
                        local[i][n] = self.transfer_dict[k](local[i][n] * self.beta[n][0])
                        break
                for l in range(self.L):
                    output[i][l] += local[i][n] * self.weight2[n][l]

        loss = np.sum((output - self.train_y) ** 2)

        avg_loss = loss
        return output,loss

if __name__ == '__main__':
    train_x = [[1,2,3],[4,5,6]]
    train_x = np.array(train_x)
    train_y = [[4,5,6],[1,2,3]]
    train_y = np.array(train_y)
    gvm =GVM(train_x=train_x,train_y=train_y)
    output,loss = gvm.train()
    print(output)
    print(loss)