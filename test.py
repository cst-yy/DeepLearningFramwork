"""
@Time ： 2023/7/24 8:29
@Auth ： yangyu
@File ：test.py
@Motto：ABC(Always Be Coding)
"""
import math

# nn
import numpy as np

from descalefed import datasets
from descalefed import functions as F
from descalefed import no_grad, models, optimizers, dataLoaders, cuda
import time

# MLP
# if __name__ == '__main__':
#     np.random.seed(42)
#     x = np.random.randn(100, 1)
#     y = np.random.randn(100, 1) + np.sin(2 * np.pi * x)
#
#
#     def sigmoid(x):
#         x = as_variable(x)
#         y = 1 / (1 + exp(-x))
#         return y
#
#
#     I, H, O = 1, 10, 1
#     W1 = Variable(0.01 * np.random.randn(I, H), name='W1')
#     b1 = Variable(np.zeros(H), name='b1')
#     W2 = Variable(0.01 * np.random.randn(H, O), name='W2')
#     b2 = Variable(np.zeros(O), name='b2')
#
#
#     def predict(x):
#         y = F.linear(x, W1, b1)
#         # 前向計算過程，調用np的exp
#         y = F.sigmoid(y)
#         y = F.linear(y, W2, b2)
#         return y
#
#
#     lr = 0.2
#     iters = 10000
#
#     for i in range(iters):
#         y_pred = predict(x)
#         loss = F.mean_squared_error(y_pred, y)
#         # utils.plt_dot_graph(loss, verbose=False, to_file='mean_squared_error.png')
#         W1.cleargrad()
#         b1.cleargrad()
#         W2.cleargrad()
#         b2.cleargrad()
#         loss.backward()
#
#         # 不加梯度計算 會導致反向傳播計算圖錯誤  接觸以上注釋測試計算圖
#         with no_grad():
#             W1.data -= lr * W1.grad.data
#             b1 -= lr * b1.grad.data
#             W2.data -= lr * W2.grad.data
#             b2 -= lr * b2.grad.data
#
#         if i % 1000 == 0:
#             print(loss)

# NN
# if __name__ == '__main__':
#     np.random.seed(42)
#     x = np.random.rand(100, 1)
#     y = 5 + 2 * x + np.random.rand(100, 1)
#
#     x, y = Variable(x), Variable(y)
#
#     W = Variable(np.zeros((1, 1)))
#     b = Variable(np.zeros(1))
#
#
#     def predict(x):
#         y = F.matmul(x, W) + b
#         return y
#
#
#     def mean_squared_error(x0, x1):
#         diff = x0 - x1
#         return F.sum(diff ** 2) / len(diff)
#
#
#     lr = 0.1
#     iters = 100
#
#     for i in range(iters):
#         y_pred = predict(x)
#         loss = mean_squared_error(y, y_pred)
#
#         W.cleargrad()
#         b.cleargrad()
#         loss.backward()
#
#         W.data -= lr * W.grad.data
#         b.data -= lr * b.grad.data
#         print(W, b, loss)


# models & optimizers
# if __name__ == '__main__':
#     np.random.seed(42)
#     x = np.random.rand(100, 1)
#     y = np.sin(2 * np.pi * x) + np.random.rand(100, 1)
#
#     lr = 0.1
#     max_iter = 10000
#     hidden_size = 10
#
#     model = models.MLP([hidden_size, 1])
#     optimizer = optimizers.MomentumSGD(lr)
#     # optimizer = optimizers.SGD(lr)
#     optimizer.setup(model)
#
#     for i in range(max_iter):
#         y_pred = model(x)
#         loss = F.mean_squared_error(y_pred, y)
#
#         model.cleargrads()
#         loss.backward()
#
#         with no_grad():
#             # for p in model.params():
#             #     p.data -= lr * p.grad.data
#             optimizer.update()
#
#         if i % 1000 == 0:
#             print(loss)


#             Datasets
# if __name__ == '__main__':
# train_set = datasets.Spiral(train=True)
# print(train_set[0])
# print(len(train_set))
#
# train_set = datasets.Spiral()
#
# batch_index = [0, 1, 2]
# batch = [train_set[i] for i in batch_index]
#
# x = np.array([example[0] for example in batch])
# t = np.array([example[1] for example in batch])
#
# print(x.shape)
# print(t.shape)

# —————————————————————————————————————————————————————————————————————————————————

# max_epochs = 5
# batch_size = 100
# hidden_size = 1000
# lr = 0.1
#
# # train_set = datasets.Spiral(train=True)
# # test_set = datasets.Spiral(train=False)
#
# train_set = datasets.MNIST(train=True)
# test_set = datasets.MNIST(train=False)
# train_loader = dataLoaders.DataLoader(train_set, batch_size)
# test_loader = dataLoaders.DataLoader(test_set, batch_size, shuffle=False)
#
# model = models.MLP((hidden_size, 10), activation=F.relu)
# optimizer = optimizers.SGD(lr).setup(model)
#
# data_size = len(train_set)
# max_iter = math.ceil(data_size / batch_size)
#
# # if cuda.gpu_enable:
# #     train_loader.to_gpu()
# #     model.to_gpu()
#
# for epoch in range(max_epochs):
#     index = np.random.permutation(data_size)
#     sum_loss, sum_acc = 0, 0
#
#     for x, t in train_loader:
#         y = model(x)
#         loss = F.softmax_cross_entropy(y, t)
#         acc = F.accuracy(y, t)
#         model.cleargrads()
#         loss.backward()
#         with no_grad():
#             optimizer.update()
#
#         sum_loss += float(loss.data) * len(t)
#         sum_acc += float(acc.data) * len(t)
#
#     print('________________________________________________________________')
#     print('epoch:{}'.format(epoch + 1))
#     print('train loss:{:.4f}, accuracy:{:.4f}'.format(sum_loss / len(train_set), sum_acc / len(train_set)))
#
#     sum_loss, sum_acc = 0, 0
#     with no_grad():
#         for x, t in test_loader:
#             y = model(x)
#             loss = F.softmax_cross_entropy(y, t)
#             acc = F.accuracy(y, t)
#             sum_loss += float(loss.data) * len(t)
#             sum_acc += float(acc.data) * len(t)
#     print('test loss:{:.4f}, accuracy:{:.4f}'.format(sum_loss / len(train_set), sum_acc / len(train_set)))


# gpu test
if __name__ == '__main__':
    max_epoch = 5
    batch_size = 100

    train_set = datasets.MNIST(train=True)
    train_loader = dataLoaders.DataLoader(train_set, batch_size)
    model = models.MLP((1000, 10))
    optimizer = optimizers.SGD().setup(model)

    # GPU mode
    if cuda.gpu_enable:
        train_loader.to_gpu()
        model.to_gpu()

    for epoch in range(max_epoch):
        start = time.time()
        sum_loss = 0

        for x, t in train_loader:
            y = model(x)
            loss = F.softmax_cross_entropy(y, t)
            model.cleargrads()
            loss.backward()
            optimizer.update()
            sum_loss += float(loss.data) * len(t)

        elapsed_time = time.time() - start
        print('epoch: {}, loss: {:.4f}, time: {:.4f}[sec]'.format(
            epoch + 1, sum_loss / len(train_set), elapsed_time))
