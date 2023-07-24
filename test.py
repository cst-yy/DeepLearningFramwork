"""
@Time ： 2023/7/24 8:29
@Auth ： yangyu
@File ：test.py
@Motto：ABC(Always Be Coding)
"""

# nn
import numpy as np
from descalefed import as_variable, Variable, utils, no_grad
from descalefed.functions import exp
from descalefed import functions as F

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


if __name__ == '__main__':
    np.random.seed(42)
    x = np.random.rand(100, 1)
    y = 5 + 2 * x + np.random.rand(100, 1)

    x, y = Variable(x), Variable(y)

    W = Variable(np.zeros((1, 1)))
    b = Variable(np.zeros(1))


    def predict(x):
        y = F.matmul(x, W) + b
        return y


    def mean_squared_error(x0, x1):
        diff = x0 - x1
        return F.sum(diff ** 2) / len(diff)


    lr = 0.1
    iters = 100

    for i in range(iters):
        y_pred = predict(x)
        loss = mean_squared_error(y, y_pred)

        W.cleargrad()
        b.cleargrad()
        loss.backward()

        W.data -= lr * W.grad.data
        b.data -= lr * b.grad.data
        print(W, b, loss)
