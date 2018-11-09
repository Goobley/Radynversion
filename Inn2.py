import torch
from torch import nn
import numpy as np

from FrEIA.framework import InputNode, OutputNode, Node, ReversibleGraphNet
from FrEIA.modules import rev_multiplicative_layer, F_fully_connected, permute_layer, flattening_layer, split_layer, cat_layer, glow_coupling_layer

from loss import mse, mse_tv, mmd_multiscale_on

from scipy.interpolate import interp1d

from copy import deepcopy
from itertools import accumulate
import pickle

PadOp = '!!PAD'
ZeroPadOp = '!!ZeroPadding'

def schema_min_len(schema, zeroPadding):
    length = sum(s[1] if s[0] != PadOp else 0 for s in schema) \
            + zeroPadding * (len([s for s in schema if s[0] != PadOp]) - 1)
    return length
        
class DataSchema1D:
    def __init__(self, inp, minLength, zeroPadding, zero_pad_fn=torch.zeros):
        self.zero_pad = zero_pad_fn
        # Check schema is valid
        padCount = sum(1 if i[0] == PadOp else 0 for i in inp)
        for i in range(len(inp)-1):
            if inp[i][0] == PadOp and inp[i+1][0] == PadOp:
                raise ValueError('Schema cannot contain two consecutive \'!!PAD\' instructions.')
        # if padCount > 1:
        #     raise ValueError('Schema can only contain one \'!!PAD\' instruction.')
        if len([i for i in inp if i[0] != PadOp]) > len(set([i[0] for i in inp if i[0] != PadOp])):
            raise ValueError('Schema names must be unique within a schema.')
        
        # Find length without extra padding (beyond normal channel separation)
        length = schema_min_len(inp, zeroPadding)
        if (minLength - length) // padCount != (minLength - length) / padCount:
            raise ValueError('Schema padding isn\'t divisible by number of PadOps')

        # Build schema
        schema = []
        padding = (ZeroPadOp, zeroPadding)
        for j, i in enumerate(inp):
            if i[0] == PadOp:
                if j == len(inp) - 1:
                    # Count the edge case where '!!PAD' is the last op and a spurious
                    # extra padding gets inserted before it
                    if schema[-1] == padding:
                        del schema[-1]

                if length < minLength:
                    schema.append((ZeroPadOp, (minLength - length) // padCount))
                continue

            schema.append(i)
            if j != len(inp) - 1:
                schema.append(padding)

        if padCount == 0 and length < minLength:
            schema.append((ZeroPadOp, minLength - length))
        
        # Fuse adjacent zero padding -- no rational way to have more than two in a row 
        fusedSchema = []
        i = 0
        while True:
            if i >= len(schema):
                break

            if i < len(schema) - 1  and schema[i][0] == ZeroPadOp and schema[i+1][0] == ZeroPadOp:
                fusedSchema.append((ZeroPadOp, schema[i][1] + schema[i+1][1]))
                i += 1
            else:
                fusedSchema.append(schema[i])
            i += 1
        # Also remove 0-width ZeroPadding
        fusedSchema = [s for s in fusedSchema if s != (ZeroPadOp, 0)]
        self.schema = fusedSchema
        schemaTags = [s[0] for s in self.schema if s[0] != ZeroPadOp]
        tagIndices = [0] + list(accumulate([s[1] for s in self.schema]))
        tagRange = [(s[0], range(tagIndices[i], tagIndices[i+1])) for i, s in enumerate(self.schema) if s[0] != ZeroPadOp]
        for name, r in tagRange:
            setattr(self, name, r)
        self.len = tagIndices[-1]

    def __len__(self):
        return self.len

    def fill(self, entries, zero_pad_fn=None, batchSize=None, checkBounds=False, dev='cpu'):
        # Try and infer batchSize
        if batchSize is None:
            for k, v in entries.items():
                if not callable(v):
                    batchSize = v.shape[0]
                    break
            else:
                raise ValueError('Unable to infer batchSize from entries (all fns?). Set batchSize manually.')
        
        if checkBounds:
            try:
                for s in self.schema:
                    if s[0] == ZeroPadOp:
                        continue
                    entry = entries[s[0]]
                    if not callable(entry):
                        if len(entry.shape) != 2:
                            raise ValueError('Entry: %s must be a 2D array or fn.' % s[0])
                        if entry.shape[0] != batchSize:
                            raise ValueError('Entry: %s does not match batchSize along dim=0.' % s[0]) 
                        if entry.shape[1] != s[1]:
                            raise ValueError('Entry: %s does not match schema dimension.' % s[0]) 
            except KeyError as e:
                raise ValueError('No key present in entries to schema: ' + repr(e))
         
        # Use different zero_pad if specified
        if zero_pad_fn is None:
             zero_pad_fn = self.zero_pad
        
        # Fill in the schema, throw exception if entry is missing
        reifiedSchema = []
        try:
            for s in self.schema:
                if s[0] == ZeroPadOp:
                    reifiedSchema.append(zero_pad_fn(batchSize, s[1]))
                else:
                    entry = entries[s[0]]
                    if callable(entry):
                        reifiedSchema.append(entry(batchSize, s[1]))
                    else:
                        reifiedSchema.append(entry)
        except KeyError as e:
            raise ValueError('No key present in entries to schema: ' + repr(e))

        reifiedSchema = torch.cat(reifiedSchema, dim=1)
        return reifiedSchema

    def __repr__(self):
        return repr(self.schema)


class RadynversionNet(ReversibleGraphNet):
    def __init__(self, inputs, outputs, zeroPadding=8, numInvLayers=5, dropout=0.05, minSize=None):
        # Determine dimensions and construct DataSchema
        inMinLength = schema_min_len(inputs, zeroPadding)
        outMinLength = schema_min_len(outputs, zeroPadding)
        minLength = max(inMinLength, outMinLength)
        if minSize is not None:
            minLength = max(minLength, minSize)
        self.inSchema = DataSchema1D(inputs, minLength, zeroPadding)
        self.outSchema = DataSchema1D(outputs, minLength, zeroPadding)
        if len(self.inSchema) != len(self.outSchema):
            raise ValueError('Input and output schemas do not have the same dimension.')

        # Build net graph
        inp = InputNode(len(self.inSchema), name='Input (0-pad extra channels)')
        nodes = [inp]

        for i in range(numInvLayers):
            nodes.append(Node([nodes[-1].out0], rev_multiplicative_layer,
                         {'F_class': F_fully_connected, 'clamp': 2.0,
                          'F_args': {'dropout': dropout if i != numInvLayers - 1 else 0.0}}, name='Inv%d' % i))
#             if (i != numInvLayers - 1) and (i % 2 == 0):
            if (i != numInvLayers - 1):
                nodes.append(Node([nodes[-1].out0], permute_layer, {'seed': i}, name='Permute%d' % i))

        nodes.append(OutputNode([nodes[-1].out0], name='Output'))
        # Build net
        super().__init__(nodes)


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
                        loss_fit=mse, loss_latent=None, loss_backward=None):
        if miniBatchSize is None:
            miniBatchSize = self.atmosData.batchSize

        if loss_latent is None:
            loss_latent = mmd_multiscale_on(self.dev)

        if loss_backward is None:
            loss_backward = mmd_multiscale_on(self.dev)

        decayEpochs = (numEpochs * miniBatchesPerEpoch) // metaEpoch
        gamma = 0.004**(1.0 / decayEpochs)

        # self.optim = torch.optim.Adam(self.model.parameters(), lr=lr, betas=(0.8, 0.8),
        #                               eps=1e-06, weight_decay=l2Reg)
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

        lTot = 0
        miniBatchIdx = 0
        wRevScale = min(epoch / (0.5 * self.numEpochs), 1)**3
        noiseScale = (1 - wRevScale) * self.zerosNoiseScale
        # noiseScale = self.zerosNoiseScale

        # def pad_fn(*x):
        #     # return noiseScale * torch.randn(*x)
        #     return torch.zeros(*x)
        # pad_fn = lambda *x: noiseScale * torch.randn(*x, device=self.dev)
        pad_fn = lambda *x: torch.zeros(*x, device=self.dev)
        randn = lambda *x: torch.randn(*x, device=self.dev)

        for x, y in self.atmosData.trainLoader:
            miniBatchIdx += 1

            if miniBatchIdx > self.miniBatchesPerEpoch:
                break

            x, y = x.to(self.dev), y.to(self.dev)
            yClean = y.clone()

            xp = self.model.inSchema.fill({'ne': x[:, 0], 
                                           'temperature': x[:, 1], 
                                           'vel': x[:, 2]},
                                          zero_pad_fn=pad_fn)
            yzp = self.model.outSchema.fill({'Halpha': y[:, 0], 
                                             'Ca8542': y[:, 1], 
                                             'LatentSpace': randn},
                                            zero_pad_fn=pad_fn)

            self.optim.zero_grad()

            out = self.model(xp)

            # lForward = self.wPred * (self.loss_fit(y[:, 0], out[:, self.model.outSchema.Halpha]) + 
            #                          self.loss_fit(y[:, 1], out[:, self.model.outSchema.Ca8542]))
            lForward = self.wPred * self.loss_fit(yzp[:, :self.model.outSchema.LatentSpace[0]], out[:, :self.model.outSchema.LatentSpace[0]])

            
            outLatentGradOnly = torch.cat((out[:, self.model.outSchema.Halpha].data, 
                                           out[:, self.model.outSchema.Ca8542].data, 
                                           out[:, self.model.outSchema.LatentSpace]), 
                                          dim=1)
            unpaddedTarget = torch.cat((yzp[:, self.model.outSchema.Halpha], 
                                        yzp[:, self.model.outSchema.Ca8542], 
                                        yzp[:, self.model.outSchema.LatentSpace]), 
                                       dim=1)
            
            lForward += self.wLatent * self.loss_latent(outLatentGradOnly, unpaddedTarget)
            lTot += lForward.data.item()

            lForward.backward()

            yzpRev = self.model.outSchema.fill({'Halpha': yClean[:, 0], 
                                                'Ca8542': yClean[:, 1], 
                                                'LatentSpace': out[:, self.model.outSchema.LatentSpace].data},
                                               zero_pad_fn=pad_fn)
            yzpRevRand = self.model.outSchema.fill({'Halpha': yClean[:, 0], 
                                                    'Ca8542': yClean[:, 1], 
                                                    'LatentSpace': randn},
                                                   zero_pad_fn=pad_fn)

            outRev = self.model(yzpRev, rev=True)
            outRevRand = self.model(yzpRevRand, rev=True)

            # THis guy should have been OUTREVRAND!!!
            # xBack = torch.cat((outRevRand[:, self.model.inSchema.ne],
            #                    outRevRand[:, self.model.inSchema.temperature],
            #                    outRevRand[:, self.model.inSchema.vel]),
            #                   dim=1)
            # lBackward = self.wRev * wRevScale * self.loss_backward(xBack, x.reshape(self.miniBatchSize, -1))
            lBackward = self.wRev * wRevScale * self.loss_backward(outRevRand[:, self.model.inSchema.ne[0]:self.model.inSchema.vel[-1]+1], 
                                                                   xp[:, self.model.inSchema.ne[0]:self.model.inSchema.vel[-1]+1])

            lBackward += 0.5 * self.wPred * self.loss_fit(outRev, xp)
            lTot += lBackward.data.item()

            lBackward.backward()

            for p in self.model.parameters():
                p.grad.data.clamp_(-15.0, 15.0)

            self.optim.step()

        return lTot / miniBatchIdx

    def test(self, maxBatches=10):
        self.model.eval()

        forwardError = []
        backwardError = []

        batchIdx = 0

        pad_fn = lambda *x: torch.zeros(*x, device=self.dev)
        randn = lambda *x: torch.randn(*x, device=self.dev)
        with torch.no_grad():
            for x, y in self.atmosData.testLoader:
                batchIdx += 1
                if batchIdx >= maxBatches:
                    break

                x, y = x.to(self.dev), y.to(self.dev)

                inp = self.model.inSchema.fill({'ne': x[:, 0],
                                                'temperature': x[:, 1],
                                                'vel': x[:, 2]},
                                               zero_pad_fn=pad_fn)
                inpBack = self.model.outSchema.fill({'Halpha': y[:, 0],
                                                     'Ca8542': y[:, 1],
                                                     'LatentSpace': randn},
                                                    zero_pad_fn=pad_fn)
                                                    
                out = self.model(inp)
                f = self.loss_fit(out[:, self.model.outSchema.Halpha], y[:, 0]) + \
                    self.loss_fit(out[:, self.model.outSchema.Ca8542], y[:, 1])
                forwardError.append(f)

                outBack = self.model(inpBack, rev=True)
                b = self.loss_fit(out[:, self.model.inSchema.ne], x[:, 0]) + \
                    self.loss_fit(out[:, self.model.inSchema.temperature], x[:, 1]) + \
                    self.loss_fit(out[:, self.model.inSchema.vel], x[:, 2])
                backwardError.append(b)
        
            fE = np.mean(forwardError)
            bE = np.mean(backwardError)

            return fE, bE, out, outBack


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
            
    def split_data_and_init_loaders(self, batchSize, padLines=False, linePadValue='Edge', zeroPadding=0, testingFraction=0.2):
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
        maxIdx = int(self.atmosIn.shape[0] * (1.0 - testingFraction)) + 1
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


