"""
@Time ： 2023/7/24 22:16
@Auth ： yangyu
@File ：layers.py
@Motto：ABC(Always Be Coding)
"""

from descalefed import Parameter, no_grad, cuda
import numpy as np
import weakref
from descalefed import functions as F


class Layer:
    def __init__(self):
        self._params = set()

    def __setattr__(self, key, value):
        if isinstance(value, (Parameter, Layer)):
            self._params.add(key)
        super().__setattr__(key, value)

    def __call__(self, *inputs):
        outputs = self.forward(*inputs)
        if not isinstance(outputs, tuple):
            outputs = (outputs,)
        self.inputs = [weakref.ref(input) for input in inputs]
        self.outputs = [weakref.ref(output) for output in outputs]
        return outputs if len(outputs) > 1 else outputs[0]

    def forward(self, inputs):
        raise NotImplementedError()

    def params(self):
        for name in self._params:
            obj = self.__dict__[name]

            if isinstance(obj, Layer):
                yield from obj.params()
            else:
                yield obj

    def cleargrads(self):
        for param in self.params():
            param.cleargrad()

    def to_cpu(self):
        for param in self.params():
            param.to_cpu()

    def to_gpu(self):
        for param in self.params():
            param.to_gpu()


class Linear(Layer):
    def __init__(self, out_size, nobias=False, dtype=np.float32, in_size=None):
        super().__init__()
        self.in_size = in_size
        self.out_size = out_size
        self.dtype = dtype

        self.W = Parameter(None, name='W')
        if self.in_size is not None:
            self._init_W()

        if nobias:
            self.b = None
        else:
            self.b = Parameter(np.zeros(out_size, dtype=dtype), name='b')

    def _init_W(self, xp=np):
        I, O = self.in_size, self.out_size
        W_data = xp.random.randn(I, O).astype(self.dtype) * np.sqrt(1 / I)
        self.W.data = W_data

    def forward(self, x):
        if self.W.data is None:
            self.in_size = x.shape[1]
            xp = cuda.get_array_module(x)
            self._init_W(xp)

        y = F.linear(x, self.W, self.b)
        return y

# if __name__ == '__main__':
#     # layer = Layer()
#     # layer.p1 = Parameter(np.array(1))
#     # layer.p2 = Parameter(np.array(2))
#     # layer.p3 = Variable(np.array(3))
#     # layer.p4 = 'test'
#     #
#     # print(layer.params)
#     # print('________________________________________________________________')
#     # for name in layer._params:
#     #     print(name, layer.__dict__[name])
#
#     np.random.seed(42)
#     x = np.random.rand(100, 1)
#     y = np.sin(2 * np.pi * x) + np.random.rand(100, 1)
#
#     l1 = Linear(10)
#     l2 = Linear(1)
#
#
#     def predict(x):
#         y = l1(x)
#         y = F.sigmoid(y)
#         y = l2(y)
#         return y
#
#
#     lr = 0.2
#     iters = 10000
#
#     for i in range(iters):
#         y_pred = predict(x)
#         loss = F.mean_squared_error(y_pred, y)
#
#         l1.cleargrads()
#         l2.cleargrads()
#         loss.backward()
#
# with no_grad():
#     for l in [l1, l2]:
#         for p in l.params():
#             p.data -= lr * p.grad.data
#
#         if i % 1000 == 0:
#             print(loss)
