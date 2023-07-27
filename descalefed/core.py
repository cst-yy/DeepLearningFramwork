"""
@Time ： 2023/7/18 20:51
@Auth ： yangyu
@File ：core_simple.py
@Motto：ABC(Always Be Coding)
"""
import contextlib
import weakref
import descalefed
import numpy as np
from descalefed import cuda
try:
    import cupy
    array_types = (np.ndarray, cupy.ndarray)
except ImportError:
    array_types = (np.ndarray)


def setup_variable():
    Variable.__mul__ = mul
    Variable.__rmul__ = mul
    Variable.__add__ = add
    Variable.__radd__ = add
    Variable.__neg__ = neg
    Variable.__sub__ = sub
    Variable.__rsub__ = rsub
    Variable.__truediv__ = div
    Variable.__rtruediv__ = rdiv
    Variable.__pow__ = pow
    Variable.__getitem__ = descalefed.functions.get_item


class Config:
    enable_backprop = True
    train = True


def as_array(x, array_module=np):
    if array_module.isscalar(x):
        return array_module.array(x)
    return x


def add(x0, x1):
    x1 = as_array(x1, cuda.get_array_module(x0))
    return AddFunction()(x0, x1)


def no_grad():
    return using_config('enable_backprop', False)


def mul(x0, x1):
    x1 = as_array(x1, cuda.get_array_module(x0))
    return Mul()(x0, x1)


def neg(x):
    f = Neg()
    return f(x)


class Variable:
    __array_priority__ = 200

    def __init__(self, data, name=None):
        data = as_array(data)

        if data is not None:
            if not isinstance(data, array_types):
                raise TypeError(f'{type(data)} is not supported')

        self.data = data
        self.name = name
        self.grad = None
        self.creator = None
        self.generation = 0

    @property
    def shape(self):
        return self.data.shape

    @property
    def dtype(self):
        return self.data.dtype

    @property
    def ndim(self):
        return self.data.ndim

    @property
    def size(self):
        return self.data.size

    def __len__(self):
        return len(self.data)

    def __repr__(self):
        if self.data is None:
            return 'Variable(None)'
        p = str(self.data).replace('\n', '\n' + ' ' * 9)
        return f'Variable({p})'

    def __mul__(self, other):
        return mul(self, other)

    def set_creator(self, func):
        self.creator = func
        self.generation = func.generation + 1

    def backward(self, retain_grad=False, create_graph=False):
        if self.grad is None:
            xp = cuda.get_array_module(self.data)
            self.grad = Variable(xp.ones_like(self.data))

        funcs = []
        seen_set = set()

        def add_func(f):
            if f not in seen_set:
                funcs.append(f)
                seen_set.add(f)
                funcs.sort(key=lambda x: x.generation)

        add_func(self.creator)

        while funcs:
            f = funcs.pop()
            gys = [output().grad for output in f.outputs]

            with using_config('enable_backprop', create_graph):
                gxs = f.backward(*gys)
                if not isinstance(gxs, tuple):
                    gxs = (gxs,)

                for x, gx in zip(f.inputs, gxs):
                    if x.grad is None:
                        x.grad = gx
                    else:
                        x.grad = x.grad + gx

                    if x.creator is not None:
                        add_func(x.creator)

            if not retain_grad:
                for y in f.outputs:
                    y().grad = None

    def cleargrad(self):
        self.grad = None

    def reshape(self, *shape):
        if len(shape) == 1 and isinstance(shape[0], (tuple, list)):
            shape = shape[0]
        return descalefed.functions.reshape(self, shape)

    def transpose(self):
        # todo: 支持顺序索引的变换
        return descalefed.functions.transpose(self)

    # 不用加括号调用此属性 加注释
    @property
    def T(self):
        return descalefed.functions.transpose(self)

    def sum(self, axis=None, keepdims=False):
        return descalefed.functions.sum(self, axis, keepdims)

    def to_cpu(self):
        if self.data is not None:
            self.data = cuda.as_numpy(self.data)

    def to_gpu(self):
        if self.data is not None:
            self.data = cuda.as_cupy(self.data)


def as_variable(obj):
    if isinstance(obj, Variable):
        return obj
    return Variable(obj)


class Function:
    def __call__(self, *inputs):
        inputs = [as_variable(x) for x in inputs]

        xs = [x.data for x in inputs]
        ys = self.forward(*xs)
        if not isinstance(ys, tuple):
            ys = (ys,)

        outputs = [Variable(as_array(y)) for y in ys]

        if Config.enable_backprop:
            self.generation = max([x.generation for x in inputs])
            for output in outputs:
                output.set_creator(self)
            self.inputs = inputs
            self.outputs = [weakref.ref(output) for output in outputs]

        return outputs if len(outputs) > 1 else outputs[0]

    def forward(self, xs):
        raise NotImplementedError

    def backward(self, grad):
        raise NotImplementedError


class Neg(Function):
    def forward(self, x):
        return -x

    def backward(self, gy):
        return -gy


class Sub(Function):
    def forward(self, x0, x1):
        self.x0_shape = x0.shape
        self.x1_shape = x1.shape
        y = x0 - x1
        return y

    def backward(self, gy):
        gx0, gx1 = gy, -gy

        if self.x0_shape != self.x1_shape:
            gx0 = descalefed.functions.sum_to(gx0, self.x0_shape)
            gx1 = descalefed.functions.sum_to(gx1, self.x1_shape)
        return gx0, gx1


def sub(x0, x1):
    x1 = as_array(x1, cuda.get_array_module(x0))
    return Sub()(x0, x1)


def rsub(x0, x1):
    x1 = as_array(x1, cuda.get_array_module(x0))
    return Sub()(x1, x0)


class AddFunction(Function):
    def forward(self, x0, x1):
        self.x0_shape = x0.shape
        self.x1_shape = x1.shape
        y = x0 + x1
        return y

    def backward(self, gy):
        gx0, gx1 = gy, gy
        if self.x0_shape != self.x1_shape:
            gx0 = descalefed.functions.sum_to(gx0, self.x0_shape)
            gx1 = descalefed.functions.sum_to(gx1, self.x1_shape)
        return gx0, gx1


@contextlib.contextmanager
def using_config(name, value):
    old_value = getattr(Config, name)
    setattr(Config, name, value)
    try:
        yield
    finally:
        setattr(Config, name, old_value)


def test_model():
    return using_config('train', False)



class Mul(Function):
    def forward(self, x0, x1):
        self.x0_shape = x0.shape
        self.x1_shape = x1.shape
        y = x0 * x1
        return y

    def backward(self, gy):
        x0, x1 = self.inputs
        gx0 = gy * x1
        gx1 = gy * x0

        if self.x0_shape != self.x1_shape:
            gx0 = descalefed.functions.sum_to(gx0, self.x0_shape)
            gx1 = descalefed.functions.sum_to(gx1, self.x1_shape)
        return gx0, gx1


class Div(Function):
    def forward(self, x0, x1):
        self.x0_shape = x0.shape
        self.x1_shape = x1.shape
        y = x0 / x1
        return y

    def backward(self, gy):
        x0, x1 = self.inputs
        gx0 = gy / x1
        gx1 = gy * (-x0 / x1 ** 2)

        if self.x0_shape != self.x1_shape:
            gx0 = descalefed.functions.sum_to(gx0, self.x0_shape)
            gx1 = descalefed.functions.sum_to(gx1, self.x1_shape)
        return gx0, gx1


def div(x0, x1):
    x1 = as_array(x1, cuda.get_array_module(x0))
    return Div()(x0, x1)


def rdiv(x0, x1):
    x1 = as_array(x1, cuda.get_array_module(x0))
    return Div()(x1, x0)


class Pow(Function):
    def __init__(self, power):
        power = power if isinstance(power, Variable) else Variable(power)
        self.power = power.data

    def forward(self, x):
        return x ** self.power

    def backward(self, gy):
        x = self.inputs[0]
        c = self.power
        return gy * c * x ** (c - 1)


def pow(x, c):
    return Pow(c)(x)

class Parameter(Variable):
    pass