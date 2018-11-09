import torch
from torch import nn
import numpy as np

from FrEIA.framework import InputNode, OutputNode, Node, ReversibleGraphNet
from FrEIA.modules import rev_multiplicative_layer, F_fully_connected, permute_layer, flattening_layer, split_layer, cat_layer, glow_coupling_layer

from loss import mse, mse_tv, mmd_multiscale_on

from scipy.interpolate import interp1d

from copy import deepcopy
import pickle


class stacking_layer(nn.Module):
    '''Squishes a 1-D tensor into an N-D tensor by reshape'''
    def __init__(self, dims_in, num_chan, dim=0):
        super().__init__()
        assert len(dims_in) == 1, "Stacking layer takes exactly one input tensor"
        split_size = dims_in[0][dim] // num_chan
        if isinstance(split_size, int):
            assert dims_in[0][dim] % split_size == 0, (
                "Tensor size not divisible by split size"
            )
        else:
            assert False, 'split_size must be an int'
        self.split_size = split_size
        self.num_chan = num_chan
        self.dim = dim

    def forward(self, x, rev=False):
        if rev:
            # return [torch.cat(x, dim=self.dim+1)]
            return [x[0].view(x[0].shape[0], -1)]
        else:
            # return [torch.stack(torch.split(x[0], self.split_size,
            #                    dim=self.dim+1), dim=self.dim+1)]
            return [x[0].view(x[0].shape[0], self.num_chan, self.split_size)]

    def jacobian(self, x, rev=False):
        # TODO batch size
        return 0

    def output_dims(self, input_dims):
        assert len(input_dims) == 1, ("Stacking layer takes exactly one input "
                                      "tensor")
        output_dims = [(input_dims[0][0] // self.split_size, self.split_size)]
        return output_dims


class RadynversionNet(ReversibleGraphNet):
    def __init__(self, inChannels, inSize, outChannels, outSize, zeroPadding=8, numInvLayers=5, latentChannels=None, latentSize=None, dropout=0.1):
        if latentChannels is None and latentSize is None:
            latentChannels = inChannels
            latentSize = inSize
        elif latentChannels is None or latentSize is None:
            raise ValueError('latentChannels and latentSize must both be set if one is set')

        if not bool(numInvLayers & 1):
            raise ValueError('numInvLayers must be odd')

        channelSize = max(inSize, outSize, latentSize)
        numChannels = max(inChannels, outChannels + latentChannels)

        inp = InputNode(numChannels * (channelSize + zeroPadding), name='Input (0-pad extra channels)')
        nodes = [inp]

        for i in range(numInvLayers):
            nodes.append(Node([nodes[-1].out0], rev_multiplicative_layer,
                         {'F_class': F_fully_connected, 'clamp': 2.5,
                          'F_args': {'dropout': dropout if (i != numInvLayers - 1) and (i != 0) else 0.0}}, name='Inv%d' % i))
            if i != numInvLayers - 1:
                nodes.append(Node([nodes[-1].out0], permute_layer, {'seed': i}, name='Permute%d' % i))

        nodes.append(OutputNode([nodes[-1].out0], name='Output'))
        super().__init__(nodes)

        self.totChannels = numChannels
        self.channelSize = channelSize
        self.zeroPadding = zeroPadding
        self.numXChannels = inChannels
        self.numYChannels = outChannels
        self.numZChannels = latentChannels


class RadynversionTrainer:
    def __init__(self, model, atmosData, dev):
        self.model = model
        self.atmosData = atmosData
        self.dev = dev

        for mod_list in model.children():
            for block in mod_list.children():
                for coeff in block.children():
                    coeff.fc3.weight.data = 1e-3*torch.randn(coeff.fc3.weight.shape)

        self.model.to(dev)

    def training_params(self, numEpochs, lr=2e-3, miniBatchesPerEpoch=20, metaEpoch=12, miniBatchSize=None, 
                        l2Reg=2e-5, wPred=1500, wLatent=300, wRev=500, zerosNoiseScale=5e-3, 
                        loss_fit=mse_tv, loss_latent=None, loss_backward=None):
        if miniBatchSize is None:
            miniBatchSize = self.atmosData.batchSize

        if loss_latent is None:
            loss_latent = mmd_multiscale_on(self.dev)

        if loss_backward is None:
            loss_backward = mmd_multiscale_on(self.dev)

        decayEpochs = (numEpochs * miniBatchesPerEpoch) // metaEpoch
        gamma = 0.002**(1.0 / decayEpochs)

        self.optim = torch.optim.Adam(self.model.parameters(), lr=lr, betas=(0.8, 0.8),
                                      eps=1e-06, weight_decay=l2Reg)
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optim,
                                                         step_size=metaEpoch,
                                                         gamma=gamma)
        self.wPred = wPred
        self.wLatent = wLatent
        self.wRev = wRev
        self.zerosNoiseScale = zerosNoiseScale
        self.miniBatchSize = miniBatchSize
        self.miniBatchesPerEpoch = miniBatchesPerEpoch
        self.numEpochs = numEpochs
        self.loss_fit = loss_fit
        self.loss_latent = loss_latent
        self.loss_backward = loss_backward

    def train(self, epoch):
        self.model.train()
        dev = self.dev

        lTot = 0
        miniBatchIdx = 0
        wRevScale = (max(epoch / (0.1*self.numEpochs), 1))**3
        # noiseScale = 2 * self.zerosNoiseScale * (1 - max(epoch / self.numEpochs, 1))**3
        noiseScale = self.zerosNoiseScale

        for x, y in self.atmosData.trainLoader:
            miniBatchIdx += 1

            if miniBatchIdx > self.miniBatchesPerEpoch:
                break

            x, y = x.to(dev), y.to(dev)
            xp = torch.cat((x, noiseScale * torch.randn(self.miniBatchSize, self.model.numXChannels, self.model.zeroPadding, device=dev)), dim=2)
            yp = torch.cat((y, noiseScale * torch.randn(self.miniBatchSize, self.model.numYChannels, self.model.zeroPadding, device=dev)), dim=2)
            xp = torch.reshape(xp, (self.miniBatchSize, -1))
            yp = torch.reshape(yp, (self.miniBatchSize, -1))
            yClean = yp.clone()

            # xPadding = noiseScale * torch.randn(self.miniBatchSize, self.model.totChannels - self.model.numXChannels, self.model.channelSize + self.model.zeroPadding, device=dev)
            xPadding = noiseScale * torch.randn(self.miniBatchSize, (self.model.totChannels - self.model.numXChannels) * (self.model.channelSize + self.model.zeroPadding), device=dev)

            # if self.model.numYChannels + self.model.numZChannels < self.model.totChannels:
            #     yzPadding = noiseScale * torch.randn(self.miniBatchSize, self.model.totChannels - self.model.numYChannels - self.model.numZChannels, self.model.channelSize + self.model.zeroPadding, device=dev)
            #     yzPadded = torch.cat((y, yzPadding, torch.randn(self.miniBatchSize, self.model.numZChannels, self.model.channelSize + self.model.zeroPadding, device=dev)), dim=1)
            #     yz = torch.cat((y, yzPadded[:, -self.model.numZChannels:, :]), dim=1)
            # else:
                # yzPadded = torch.cat((y, torch.randn(self.miniBatchSize, self.model.numZChannels, self.model.channelSize + self.model.zeroPadding, device=dev)), dim=1)
                # yz = yzPadded.clone()
            yz = torch.cat([yp] + [torch.randn(self.miniBatchSize, self.model.channelSize, device=dev), noiseScale * torch.randn(self.miniBatchSize, self.model.zeroPadding, device=dev)] * self.model.numZChannels, dim=1)
            yzPadded = yz.clone()

            # yzPadded = torch.reshape(yzPadded, (self.miniBatchSize, -1))
            # yz = torch.reshape(yz, (self.miniBatchSize, -1))

            xPadded = torch.cat((xp, xPadding), dim=1)
            # xPadded = torch.reshape(xPadded, (self.miniBatchSize, -1))

            self.optim.zero_grad()

            out = self.model(xPadded)
            out = torch.reshape(out, (self.miniBatchSize, self.model.totChannels, -1))

            lForward = self.wPred * self.loss_fit(y.reshape(self.miniBatchSize, -1), out[:, :self.model.numYChannels, :-self.model.zeroPadding].reshape(self.miniBatchSize, -1))

            outputLatentGradient = torch.cat((out[:, :self.model.numYChannels, :].data, out[:, -self.model.numZChannels:, :]), dim=1)
            lForward += self.wLatent * self.loss_latent(yz.reshape((self.miniBatchSize, -1)), outputLatentGradient.reshape((self.miniBatchSize, -1)))

            lTot += lForward.data.item()
            # lForward.backward(retain_graph=True)
            lForward.backward()

            # if self.model.numYChannels + self.model.numZChannels < self.model.totChannels:
            #     yzPad = noiseScale * torch.randn(self.miniBatchSize, self.model.totChannels - self.model.numYChannels - self.model.numZChannels, self.model.channelSize + self.model.zeroPadding, device=dev)
            #     yzRevRand = torch.cat((yClean, yzPad, torch.randn(self.miniBatchSize, self.model.numZChannels, self.model.channelSize + self.model.zeroPadding, device=dev)), dim=1)
            #     yzRev = torch.cat((yClean, yzPad, out[:, -self.model.numZChannels:, :].data), dim=1)
            # else:
            #     yzRevRand = torch.cat((yClean, torch.randn(self.miniBatchSize, self.model.numZChannels, self.model.channelSize + self.model.zeroPadding, device=dev)), dim=1)
            #     yzRev = torch.cat((yClean, out[:, -self.model.numZChannels:, :].data), dim=1)
            yzRevRand = torch.cat([yClean] + [torch.randn(self.miniBatchSize, self.model.channelSize, device=dev), noiseScale * torch.randn(self.miniBatchSize, self.model.zeroPadding, device=dev)] * self.model.numZChannels, dim=1)
            yzRev = torch.cat((yClean.reshape(self.miniBatchSize, self.model.numYChannels, -1), out[:, -self.model.numZChannels:, :].data), dim=1)

            yzRev = torch.reshape(yzRev, (self.miniBatchSize, -1))
            yzRevRand = torch.reshape(yzRevRand, (self.miniBatchSize, -1))

            outRev = self.model(yzRev, rev=True)
            outRevRand = self.model(yzRevRand, rev=True)
            orr = torch.reshape(outRevRand, (self.miniBatchSize, self.model.totChannels, -1))

            lRev = self.wRev * wRevScale * self.loss_backward(orr[:, :self.model.numXChannels, :-self.model.zeroPadding].reshape((self.miniBatchSize, -1)), x.reshape((self.miniBatchSize, -1)))
            # lRev += self.wPred * self.loss_fit(outRev.reshape((self.miniBatchSize, self.model.totChannels, -1)), xPadded.reshape(self.miniBatchSize, self.model.totChannels, -1))
            lRev += self.wPred * self.loss_fit(outRev, xPadded)

            lTot += lRev.data.item()
            lRev.backward()

            for p in self.model.parameters():
                p.grad.data.clamp_(-15.0, 15.0)

            self.optim.step()

        return lTot / miniBatchIdx

    def test(self, maxBatches=10):
        dev = self.dev
        self.model.eval()

        forwardMse = []
        backwardMse = []

        batchIdx = 0
        with torch.no_grad():
            for x, y in self.atmosData.testLoader:
                batchIdx += 1
                if batchIdx >= maxBatches:
                    break
                
                x, y = x.to(dev), y.to(dev)

                xPadding = torch.zeros(self.miniBatchSize, self.model.totChannels - self.model.numXChannels, self.model.channelSize + self.model.zeroPadding, device=dev)

                if self.model.numYChannels + self.model.numZChannels < self.model.totChannels:
                    yzPadding = torch.zeros(self.miniBatchSize, self.model.totChannels - self.model.numYChannels - self.model.numZChannels, self.model.channelSize + self.model.zeroPadding, device=dev)
                    yzPadded = torch.cat((y, yzPadding, torch.randn(self.miniBatchSize, self.model.numZChannels, self.model.channelSize + self.model.zeroPadding, device=dev)), dim=1)
                else:
                    yzPadded = torch.cat((y, torch.randn(self.miniBatchSize, self.model.numZChannels, self.model.channelSize + self.model.zeroPadding, device=dev)), dim=1)

                xPadded = torch.cat((x, xPadding), dim=1)
                xPadded = torch.reshape(xPadded, (self.miniBatchSize, -1))
                yzPadded = torch.reshape(yzPadded, (self.miniBatchSize, -1))

                forward = self.model(xPadded)
                forward = torch.reshape(forward, (self.miniBatchSize, self.model.totChannels, -1))
                forwardMse.append(self.loss_fit(forward[:, :self.model.numYChannels, :], y))

                backward = self.model(yzPadded, rev=True)
                backward = torch.reshape(backward, (self.miniBatchSize, self.model.totChannels, -1))
                backwardMse.append(self.loss_fit(backward[:, :self.model.numXChannels, :], x))

        fMse = np.mean(forwardMse)
        bMse = np.mean(backwardMse)

        return fMse, bMse

class AtmosData:
    def __init__(self, dataLocations, resampleWl='ProfileLength'):
        if type(dataLocations) is str:
            dataLocations = [dataLocations]

        with open(dataLocations[0], 'rb') as p:
            data = pickle.load(p)

        if len(dataLocations) > 1:
            for dataLocation in dataLocations[1:]:
                with open(dataLocation, 'rb') as p:
                    d = pickle.load(p)

                for k in data.keys():
                    if k == 'wavelength' or k == 'z' or k == 'lineInfo':
                        continue
                    if k == 'line':
                        for i in range(len(data['line'])):
                            data[k][i] += d[k][i]
                    else:
                        try:
                            data[k] += d[k]
                        except KeyError:
                            pass

        self.temperature = torch.stack(data['temperature']).float().log10_()
        self.ne = torch.stack(data['ne']).float().log10_()
        vel = torch.stack(data['vel']).float() / 1e5
        velSign = vel / vel.abs()
        velSign[velSign != velSign] = 0
        self.vel = velSign * (vel.abs() + 1).log10()

        if resampleWl == 'ProfileLength':
            resampleWl = self.ne.shape[1]

        wls = [wl.float() for wl in data['wavelength']] 

        if resampleWl is not None:
            wlResample = [torch.from_numpy(np.linspace(torch.min(wl), torch.max(wl), num=resampleWl, dtype=np.float32)) for wl in wls]
            lineResample = []
            for lineIdx in range(len(data['lineInfo'])):
                lineProfile = []
                for line in data['line'][lineIdx]:
                    interp = interp1d(wls[lineIdx], line, assume_sorted=True, kind='cubic')
                    lineProfile.append(torch.from_numpy(interp(wlResample[lineIdx])).float())
                lineResample.append(lineProfile)
                
            lines = [torch.stack(l).float() for l in lineResample]
        else:
            wlResample = wls
            lines = [torch.stack(data['line'][idx]).float() for idx in range(len(wls))]

        self.wls = wlResample
        self.lines = lines
            
        # use the [0] the chuck the index vector away
        self.lines = [l / torch.max(l, 1, keepdim=True)[0] for l in self.lines]
        self.z = data['z'].float()
            
    def split_data_and_init_loaders(self, batchSize, padLines=False, linePadValue='Edge', zeroPadding=0):
        self.atmosIn = torch.stack([self.ne, self.temperature, self.vel]).permute(1, 0, 2)
        self.batchSize = batchSize

        if padLines and linePadValue == 'Edge':
            lPad0Size = (self.ne.shape[1] - self.lines[0].shape[1]) // 2
            rPad0Size = self.ne.shape[1] - self.lines[0].shape[1] - lPad0Size
            lPad1Size = (self.ne.shape[1] - self.lines[1].shape[1]) // 2
            rPad1Size = self.ne.shape[1] - self.lines[1].shape[1] - lPad1Size
            if any(np.array([lPad0Size, rPad0Size, lPad1Size, rPad1Size]) <= 0):
                raise ValueError('Cannot pad lines as they are already bigger than/same size as the profiles!')
            lPad0 = torch.ones(self.lines[0].shape[0], lPad0Size) * self.lines[0][:, 0].unsqueeze(1)
            rPad0 = torch.ones(self.lines[0].shape[0], rPad0Size) * self.lines[0][:, -1].unsqueeze(1)
            lPad1 = torch.ones(self.lines[1].shape[0], lPad1Size) * self.lines[1][:, 0].unsqueeze(1)
            rPad1 = torch.ones(self.lines[1].shape[0], rPad1Size) * self.lines[1][:, -1].unsqueeze(1)

            self.lineOut = torch.stack([torch.cat((lPad0, self.lines[0], rPad0), dim=1), torch.cat((lPad1, self.lines[1], rPad1), dim=1)]).permute(1, 0, 2)
        elif padLines:
            lPad0Size = (self.ne.shape[1] - self.lines[0].shape[1]) // 2
            rPad0Size = self.ne.shape[1] - self.lines[0].shape[1] - lPad0Size
            lPad1Size = (self.ne.shape[1] - self.lines[1].shape[1]) // 2
            rPad1Size = self.ne.shape[1] - self.lines[1].shape[1] - lPad1Size
            if any(np.array([lPad0Size, rPad0Size, lPad1Size, rPad1Size]) <= 0):
                raise ValueError('Cannot pad lines as they are already bigger than/same size as the profiles!')
            lPad0 = torch.ones(self.lines[0].shape[0], lPad0Size) * linePadValue
            rPad0 = torch.ones(self.lines[0].shape[0], rPad0Size) * linePadValue
            lPad1 = torch.ones(self.lines[1].shape[0], lPad1Size) * linePadValue
            rPad1 = torch.ones(self.lines[1].shape[0], rPad1Size) * linePadValue

            self.lineOut = torch.stack([torch.cat((lPad0, self.lines[0], rPad0), dim=1), torch.cat((lPad1, self.lines[1], rPad1), dim=1)]).permute(1, 0, 2)
        else:
            self.lineOut = torch.stack([self.lines[0], self.lines[1]]).permute(1, 0, 2)

        indices = np.arange(self.atmosIn.shape[0])
        np.random.shuffle(indices)

        # split off 20% for testing
        maxIdx = int(self.atmosIn.shape[0] * 0.8) + 1
        if zeroPadding != 0:
            trainIn = torch.cat((self.atmosIn[indices][:maxIdx], torch.zeros(maxIdx, self.atmosIn.shape[1], zeroPadding)), dim=2)
            trainOut = torch.cat((self.lineOut[indices][:maxIdx], torch.zeros(maxIdx, self.lineOut.shape[1], zeroPadding)), dim=2)
            testIn = torch.cat((self.atmosIn[indices][maxIdx:], torch.zeros(self.atmosIn.shape[0] - maxIdx, self.atmosIn.shape[1], zeroPadding)), dim=2)
            testOut = torch.cat((self.lineOut[indices][maxIdx:], torch.zeros(self.atmosIn.shape[0] - maxIdx, self.lineOut.shape[1], zeroPadding)), dim=2)
        else:
            trainIn = self.atmosIn[indices][:maxIdx]
            trainOut = self.lineOut[indices][:maxIdx]
            testIn = self.atmosIn[indices][maxIdx:]
            testOut = self.lineOut[indices][maxIdx:]

        self.testLoader = torch.utils.data.DataLoader(
                    torch.utils.data.TensorDataset(testIn, testOut), 
                    batch_size=batchSize, shuffle=True, drop_last=True)
        self.trainLoader = torch.utils.data.DataLoader(
                    torch.utils.data.TensorDataset(trainIn, trainOut), 
                    batch_size=batchSize, shuffle=True, drop_last=True)


