import torch

def mse(inp, target):
    return torch.mean((inp - target)**2)

def tv_chan(inp, target):
    l = inp.shape[-1]-1
    tvIn = torch.sum(torch.abs(inp[:, :, 1:] - inp[:, :, :-1]), dim=2)
    tvTarget = torch.sum(torch.abs(target[:, :, 1:] - target[:, :, :-1]), dim=2)
    return torch.mean(torch.abs(tvTarget - tvIn)) / l

def tv_no_chan(inp, target):
    l = inp.shape[-1]-1
    tvIn = torch.sum(torch.abs(inp[:, 1:] - inp[:, :-1]), dim=1)
    tvTarget = torch.sum(torch.abs(target[:, 1:] - target[:, :-1]), dim=1)
    return torch.mean(torch.abs(tvTarget - tvIn)) / l

def mse_tv(inp, target):
    return 0.9 * mse(inp, target) + 0.1 * tv_no_chan(inp, target)
    

def mmd_multiscale_on(dev):
    def mmd_multiscale(x, y):
        xx, yy, zz = torch.mm(x,x.t()), torch.mm(y,y.t()), torch.mm(x,y.t())

        rx = (xx.diag().unsqueeze(0).expand_as(xx))
        ry = (yy.diag().unsqueeze(0).expand_as(yy))

        dxx = rx.t() + rx - 2.*xx
        dyy = ry.t() + ry - 2.*yy
        dxy = rx.t() + ry - 2.*zz

        XX, YY, XY = (torch.zeros(xx.shape).to(dev),

                      torch.zeros(xx.shape).to(dev),
                      torch.zeros(xx.shape).to(dev))

#         for a in [0.2, 0.5, 0.9, 1.3]:
        for a in [0.05, 0.125, 0.225, 0.325]:
            XX += a**2 * (a**2 + dxx)**-1
            YY += a**2 * (a**2 + dyy)**-1
            XY += a**2 * (a**2 + dxy)**-1

        return torch.mean(XX + YY - 2.*XY)
    return mmd_multiscale