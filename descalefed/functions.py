"""
@Time ： 2023/7/18 20:53
@Auth ： yangyu
@File ：functions.py
@Motto：ABC(Always Be Coding)
"""
import numpy as np
from descalefed.core import Function
from descalefed.core import Variable
import matplotlib.pyplot as plt
from descalefed.utils import plt_dot_graph


class Square(Function):
    def forward(self, x):
        return x ** 2

    def backward(self, gy):
        # x = self.inputs[0].data
        x = self.inputs[0]
        gx = 2 * x * gy
        return gx


class Exp(Function):
    def forward(self, x):
        return np.exp(x)

    def backward(self, gy):
        return gy * np.exp(self.inputs)


def square(x):
    f = Square()
    return f(x)


def exp(x):
    f = Exp()
    return f(x)


class Sin(Function):
    def forward(self, x):
        return np.sin(x)

    def backward(self, gy):
        x, = self.inputs
        return gy * cos(x)


def sin(x):
    f = Sin()
    return f(x)


class Cos(Function):
    def forward(self, x):
        return np.cos(x)

    def backward(self, gy):
        x, = self.inputs
        return -gy * sin(x)


def cos(x):
    f = Cos()
    return f(x)

class Tanh(Function):
    def forward(self, x):
        return np.tanh(x)
    def backward(self, gy):
        y = self.outputs[0]()
        gx = gy * (1-y*y)
        return gx

def tanh(x):
    f = Tanh()
    return f(x)


if __name__ == '__main__':
    x = Variable(np.array(1.0))
    y = tanh(x)
    x.name = 'x'
    y.name = 'y'
    y.backward(create_graph=True)
    iters = 3

    for i in range(iters):
        gx = x.grad
        x.cleargrad()
        gx.backward(create_graph=True)

    gx = x.grad
    gx.name = 'gx'+str(iters+1)
    plt_dot_graph(gx, verbose=False, to_file='tanh.png')


# if __name__ == '__main__':
#     x = Variable(np.linspace(-7, 7, 200))
#     y = sin(x)
#     y.backward(create_graph=True)
#
#     logs = [y.data]
#
#     for i in range(3):
#         logs.append(x.grad.data)
#         gx = x.grad
#         x.cleargrad()
#         gx.backward(create_graph=True)
#
#     #     plot
#     labels = ["y = sin(x)", "y'", "y''", "y'''"]
#     for i, v in enumerate(logs):
#         plt.plot(x.data, logs[i], label=labels[i])
#     plt.legend(loc='lower right')
#     plt.show()
