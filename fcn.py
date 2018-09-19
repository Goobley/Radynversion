import torch.nn as nn

def FCN(inDim,nLayers,outDim,hiddenDim):
    '''
    A function that defines the fully-connected network that will be used as the complex function for the affine coefficients in the affine coupling layers.

    Parameters
    ----------
    inDim : int
        The dimensionality of the input to the network.
    nLayers: int
        The number of fully-connected layers to stack.
    outDim : int
        The dimensionality of the output.
    hiddenDim : int
        The dimensionality of the hidden layers in the network.
    '''

    layers = []
    layers.append(nn.Linear(inDim,hiddenDim))
    layers.append(nn.LeakyReLU(inplace=True))
    for i in range(nLayers-2):
        layers.append(nn.Linear(hiddenDim,hiddenDim))
        layers.append(nn.LeakyReLU(inplace=True))
    layers.append(nn.Linear(hiddenDim,outDim))
    layers.append(nn.LeakyReLU(inplace=True))

    return nn.Sequential(*layers)