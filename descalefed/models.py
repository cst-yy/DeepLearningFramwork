"""
@Time ： 2023/7/25 11:08
@Auth ： yangyu
@File ：models.py
@Motto：ABC(Always Be Coding)
"""

from descalefed import Layer, utils
from descalefed.layers import Linear
import descalefed.functions as F
import descalefed.layers as L


class Model(Layer):
    def plot(self, *inputs, to_file='model.png'):
        y = self.forward(*inputs)
        return utils.plt_dot_graph(y, verbose=True, to_file=to_file)


class MLP(Model):
    def __init__(self, fc_output_size, activation=F.sigmoid):
        super().__init__()
        self.activation = activation
        self.layers = []

        for i, out_size in enumerate(fc_output_size):
            layer = L.Linear(out_size)
            setattr(self, 'layers'+str(i), layer)
            self.layers.append(layer)

    def forward(self, x):
        for l in self.layers[:-1]:
            x = self.activation(l(x))
        return self.layers[-1](x)

