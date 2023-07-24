"""
@Time ： 2023/7/18 20:53
@Auth ： yangyu
@File ：functions.py
@Motto：ABC(Always Be Coding)
"""
import numpy as np
from descalefed import utils
from descalefed.core import Function
from descalefed.core import Variable, as_variable
from descalefed.utils import plt_dot_graph,reshape_sum_backward, sum_to


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
        gx = gy * (1 - y * y)
        return gx


def tanh(x):
    f = Tanh()
    return f(x)


class Reshape(Function):
    def __init__(self, shape):
        self.shape = shape

    def forward(self, x):
        self.x_shape = x.shape
        y = x.reshape(self.shape)
        return y

    def backward(self, gy):
        return reshape(gy, self.x_shape)


def reshape(x, shape):
    if x.shape == shape:
        return as_variable(x)

    return Reshape(shape)(x)


class Transpose(Function):
    def forward(self, x):
        return x.transpose()

    def backward(self, gy):
        return transpose(gy)


def transpose(x):
    return Transpose()(x)


class Sum(Function):
    def __init__(self, axis, keepdims):
        self.axis = axis
        # true会保留轴的数量 1个轴为(1,1)
        self.keepdims = keepdims

    def forward(self, x):
        self.x_shape = x.shape
        y = x.sum(axis=self.axis, keepdims=self.keepdims)
        return y

    def backward(self, gy):
        gy = reshape_sum_backward(gy, self.x_shape, self.axis, self.keepdims)

        gx = broadcast_to(gy, self.x_shape)
        return gx


def sum(x, axis=None, keepdims=False):
    return Sum(axis, keepdims)(x)


def broadcast_to(x, shape):
    if x.shape == shape:
        return as_variable(x)
    return BroadcastTo(shape)(x)

class BroadcastTo(Function):
    def __init__(self, shape):
        self.shape = shape

    def forward(self, x):
        self.x_shape = x.shape
        y = np.broadcast_to(x, self.shape)
        return y

    def backward(self, gy):
        gx = sum_to(gy, self.x_shape)
        return gx

def broadcast_to(x, shape):
    if x.shape == shape:
        return as_variable(x)
    return BroadcastTo(shape)(x)

class SumTo(Function):
    def __init__(self, shape):
        self.shape = shape

    def forward(self, x):
        self.x_shape = x.shape
        y = utils.sum_to(x, self.shape)
        return y

    def backward(self, gy):
        gx = broadcast_to(gy, self.x_shape)
        return gx

def sum_to(x, shape):
    if x.shape == shape:
        return as_variable(x)
    return SumTo(shape)(x)

class MatMul(Function):
    def forward(self, x, W):
        y = x.dot(W)
        return y

    def backward(self, gy):
        x, W = self.inputs
        gx = matmul(gy, W.T)
        gW = matmul(x.T, gy)
        return gx, gW

def matmul(x,W):
    return MatMul()(x,W)

class MeanSquaredError(Function):
    def forward(self,x0, x1):
        diff = x0 - x1
        y = (diff**2).sum() / len(diff)
        return y

    def backward(self, gy):
        x0, x1 = self.inputs
        diff = x0 - x1
        gx0 = gy * diff * (2. / len(diff))
        gx1 = -gx0
        return gx0, gx1

def mean_squared_error(x0, x1):
    return MeanSquaredError()(x0, x1)

class Linear(Function):
    def forward(self, x, W, b):
        y = x.dot(W)
        if b is not None:
            y += b
        return y

    def backward(self, gy):
        x, W, b = self.inputs
        gb = None if b.data is None else sum_to(gy, b.shape)
        gx = matmul(gy, W.T)
        gW = matmul(x.T, gy)
        return gx, gW, gb


def linear(x, W, b=None):
    return Linear()(x, W, b)


def linear_simple(x,W,b=None):
    t = matmul(x,W)
    if b is None:
        return t

    y = t+b
    t.data = None
    return y

# def sigmoid_simple(x):
#     x = as_variable(x)
#     y = 1 / (1 + exp(-x))
#     return y

class Sigmoid(Function):
    def forward(self, x):
        # xp = cuda.get_array_module(xp)
        y = 1 / (1 + np.exp(-x))
        # y = 1 / (1 + xp.exp(-x))
        # y = xp.tanh(x * 0.5) * 0.5 + 0.5  # Better implementation
        return y

    def backward(self, gy):
        y = self.outputs[0]()
        gx = gy * y * (1 - y)
        return gx


def sigmoid(x):
    return Sigmoid()(x)



if __name__ == '__main__':
    x = Variable(np.array(1.0))
    y = tanh(x)
    x.name = 'x'
    y.name = 'y'
    y.backward(create_graph=True)
    iters = 1

    for i in range(iters):
        gx = x.grad
        x.cleargrad()
        gx.backward(create_graph=True)

    gx = x.grad
    gx.name = 'gx' + str(iters + 1)
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
