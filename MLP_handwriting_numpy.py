# -*- coding: utf-8 -*-
# @Time : 2022/9/7 18:14
# @Author : wcy

import numpy as np
from matplotlib import pyplot as plt
import random
import gzip
import _pickle


class Activation:
    """
    激活函数类, 包括ReLU和Softmax激活函数, 其他函数暂未实现
    """
    @classmethod
    def ReLU(cls, Z):
        A = np.zeros_like(Z)
        return np.maximum(Z, A)

    @classmethod
    def Sigmoid(cls, Z):
        return 1 / (1 + 1 / np.exp(Z))

    @classmethod
    def Softmax(cls, Z):
        Z = np.exp(Z)
        Z_sum = Z.sum(axis=1).reshape((-1, 1))
        return Z / Z_sum


class Delta:
    """
    梯度求导类
    """
    def __init__(self, func):
        """
        :param func: Activation类的函数指针
        """
        self.func = func

    def delta(self, A):
        if self.func == Activation.ReLU:
            return (A > 0).astype(float)
        elif self.func == Activation.Sigmoid:
            return A * (1 - A)


class Loss:
    """
    损失函数类
    """
    def __init__(self, y_hat, y):
        """
        :param y_hat: 预测值
        :param y: 标签值
        """
        self.y_hat = y_hat
        self.y = y
        self.length = len(y_hat)

    def cross_entropy(self):
        """
        交叉熵损失函数
        """
        y_tmp = self.y_hat[range(self.length), self.y]
        return -np.log(y_tmp)

    def cs_delta(self):
        """
        交叉熵+softmax求导
        """
        A_delta = np.array(self.y_hat)
        A_delta[range(self.length), self.y] -= 1
        return A_delta.mean(axis=0)

    def accuracy(self):
        """
        :return: 预测准确率
        """
        y_tmp = self.y_hat.argmax(axis=1)
        cmp = y_tmp.astype(dtype=self.y.dtype) == self.y
        cmp = cmp.astype(int).sum()
        return round(float(cmp) / len(self.y), 3)


class NeuralNetwork:
    """
    神经网络类
    """
    def __init__(self, layer_lst, training_data, test_data=None):
        # layer层数, 只算隐藏层和输出层
        self.nums_layer = len(layer_lst) - 1
        self.dims_layer = [(layer_lst[i + 1], layer_lst[i]) for i in range(self.nums_layer)]
        # 训练样本个数
        self.num_training = 0
        self.W = []
        self.b = []
        # 前向传播梯度
        self.forward_grad = []
        # 后向传播梯度
        self.backward_grad = []
        self.training_features, self.training_labels = training_data[0], training_data[1]
        self.num_training = len(self.training_features)
        if test_data:
            self.test_features, self.test_labels = test_data[0], test_data[1]
        else:
            self.test_features, self.test_labels = self.training_features, self.training_labels
        self.func1 = None
        self.func2 = None

    def set_activation(self, func1=Activation.ReLU, func2=Activation.Softmax):
        """
        设置激活函数
        :param func1: 隐藏层激活函数, 默认relu
        :param func2: 输出层激活函数, 默认softmax
        """
        self.func1 = func1
        self.func2 = func2

    def init_params(self):
        """
        初始化weight, bias, bp梯度
        """
        self.W = [np.random.normal(0, 0.1, layer) for layer in self.dims_layer]
        self.b = [np.zeros((1, layer[0])) for layer in self.dims_layer]
        self.backward_grad = [np.zeros((layer[0], 1)) for layer in self.dims_layer]

    def data_iter(self, batch_size):
        """
        小批量梯度下降的迭代器
        :param batch_size: 批量大小, 在trainer方法传入
        """
        indices = list(range(self.num_training))
        random.shuffle(indices)
        for i in range(0, self.num_training, batch_size):
            idx = indices[i:min(i + batch_size, self.num_training)]
            yield self.training_features[idx], self.training_labels[idx]

    def forward_propagation(self, X):
        """
        前向传播算法
        :param X: 小批量的features
        """
        self.forward_grad.append(X)
        for i in range(self.nums_layer):
            if i == self.nums_layer - 1:
                A = self.func2(self.forward_grad[i] @ self.W[i].T + self.b[i])
            else:
                A = self.func1(self.forward_grad[i] @ self.W[i].T + self.b[i])
            self.forward_grad.append(A)

    def backward_propagation(self, y):
        """
        反向传播算法
        :param y: 小批量的labels
        """
        l = Loss(self.forward_grad[-1], y)
        self.backward_grad[-1] += l.cs_delta().reshape((-1, 1))
        d = Delta(self.func1)
        for i in reversed(range(self.nums_layer - 1)):
            tmp = d.delta(self.forward_grad[i + 1]).mean(axis=0) * (self.backward_grad[i + 1] * self.W[i + 1]).sum(
                axis=0)
            tmp = tmp.reshape((-1, 1))
            self.backward_grad[i] += tmp

    def zero_grad(self):
        """
        梯度清零
        """
        self.forward_grad = []
        for i in range(self.nums_layer):
            self.backward_grad[i] = np.zeros_like(self.backward_grad[i])

    def SGD(self, lr):
        """
        梯度下降优化算法
        :param lr: 学习率, 在trainer方法传入
        """
        for i in range(self.nums_layer):
            w_grad = self.forward_grad[i].mean(axis=0) * self.backward_grad[i]
            b_grad = self.backward_grad[i].reshape((1, -1))
            self.W[i] -= lr * w_grad
            self.b[i] -= lr * b_grad

    def trainer(self, num_epochs, lr, batch_size):
        """
        训练器
        :param num_epochs: 训练的代数
        :param lr: 学习率
        :param batch_size: 批量大小
        """
        self.loss_lst, self.accuracy_lst = [], []
        self.init_params()
        for epoch in range(num_epochs):
            for X, y in self.data_iter(batch_size):
                self.zero_grad()
                self.forward_propagation(X)
                self.backward_propagation(y)
                self.SGD(lr)
                self.zero_grad()
            self.forward_propagation(self.test_features)
            l = Loss(self.forward_grad[-1], self.test_labels)
            loss = l.cross_entropy().mean()
            acc = l.accuracy()
            self.loss_lst.append(loss)
            self.accuracy_lst.append(acc)

    def plt_show(self):
        """
        matplotlib画图
        """
        plt.figure(figsize=(14, 4))
        plt.subplot(1, 2, 1)
        plt.plot(self.loss_lst, linewidth=2.2)
        plt.title('Loss', fontsize=20)
        plt.subplot(1, 2, 2)
        plt.plot(self.accuracy_lst, linewidth=2.2)
        plt.title('Accuracy', fontsize=20)
        plt.show()


def load_data(path):
    """
    加载MNIST数据集, 把每个图片展成784维的向量
    """
    f = gzip.open(path, 'rb')
    training_data, validation_data, test_data = _pickle.load(f, encoding='bytes')
    f.close()
    return [training_data, validation_data, test_data]


if __name__ == '__main__':
    # 加载数据集
    training_data, validation_data, test_data = load_data('MNIST/mnist.pkl.gz')
    
    # 超参数
    input_dims = 784
    hidden1_dims = 256
    hidden2_dims = 256
    output_dims = 10
    num_epochs = 100
    batch_size = 16
    lr = 0.03
    relu = Activation.ReLU
    softmax = Activation.Softmax
    
    # 构建神经网络并训练
    nn = NeuralNetwork([input_dims, hidden1_dims, hidden2_dims, output_dims], training_data, validation_data)
    nn.set_activation(relu, softmax)
    nn.trainer(num_epochs, lr, batch_size)
    nn.plt_show()
