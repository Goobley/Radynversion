import torch.nn as nn
from fcn import *

class Layer(nn.Module):
    def __init__(self):
        super(Layer,self).__init__()

    def forward_(self,x,obj):
        raise NotImplementedError

    def reverse_(self,x,obj):
        raise NotImplementedError

class LayerList(Layer):
    def __init__(self,layers=None):
        super(LayerList,self).__init__()
        self.layers = nn.ModuleList(layers)

    def __getitem__(self,idx):
        return self.layers[idx]

    def forward_(self,x,obj):
        for layer in self.layers:
            x,obj = layer.forward_(x,obj)
        return x,obj

    def reverse_(self,x,obj):
        for layer in reversed(self.layers):
            x,obj = layer.reverse_(x,obj)
        return x,obj