import os
os.environ['CDF_LIB'] = '/usr/local/cdf/lib'
import radynpy.RadynCdfLoader as RadynCdfLoader
import numpy as np
from scipy.signal import savgol_filter
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
import torch
import pickle
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed

# We should possibly do the extrapolation against cell centres rather than
# interface locations, but what I can see of the default radyn analysis scripts
# is that they use the interface data everywhere. Probably doesn't really
# matter for the shape of curve. Also, there can't be k interaface heights and
# cell centred heights...
tags = ['alamb', 'q', 'nq', 'outint', 'cont', 'time', 'tg1', 'ne1', 'zmu', 'z1', 'vz1', 'n1', 'jrad', 'irad']

folder = '/local1/scratch/RadynGrid/'
files = [folder + x for x in os.listdir(folder) if x.startswith('radyn_out.val3c') and not x.endswith('em')]
outputFolder = '/local0/scratch/HAlphaGridExportStatic/'

start = -6.5e6
expTrans = 3.5e8
stop = 1.05e9
# linStep = 2e5
# expNum = 500
# linStep = 10e5
totalNum = 88
expNum = totalNum // 10
linNum = totalNum - expNum
expNum += 1 # We have to remove the first entry of the exponential grid, because it's the same as the last of the linear grid, so add an extra point to account for this
# staticAltitudeGrid = np.concatenate((np.arange(start, expTrans, linStep), np.logspace(np.log10(expTrans), np.log10(stop), num=expNum))).astype(np.float32)
staticAltitudeGrid = np.concatenate((np.linspace(start, expTrans, num=linNum), np.logspace(np.log10(expTrans), np.log10(stop), num=expNum)[1:])).astype(np.float32)
# staticAltitudeGrid = savgol_filter(staticAltitudeGrid, 7, 3)
print('Interpolating onto %d spatial points' % len(staticAltitudeGrid))

def line_intensity_with_cont(data, kr, muIdx):
    if not not data.cont[kr]:
        print('line_intensity cannot compute bf intensity')
        return

    # Made some changes to the q indexing here, because it didn't seem quite right to me
    # perhaps that's idl slicing
    # yup, it includes the last element, and python doesn't
    # 1e5 is a conversion between km/s (qnorm) and cm/s (cc)
    wl = data.alamb[kr] / (data.q[0:data.nq[kr], kr] *  data.qnorm * 1e5 / data.cc + 1)
    # The 1e8 in here comes from the conversion from angstrom to cm, but the square
    # on wavelength comes from computing the derivative of nu*lambda = c to get the
    # Jacobian
    # The second part of the parens is the continuum here, normally negligible -- especially for Halpha
    intens = (data.outint[:, 1:data.nq[kr]+1, muIdx, kr] + data.outint[:, 0, muIdx, kr][:, np.newaxis]) *  data.cc * 1e8 / (wl**2)[np.newaxis, :] 
    # wl is retruned in angstrom, intens in erg/cm^2/sr/A/s
    return wl[::-1], intens[:,::-1]

def line_dict(data, export, lineIdx):

    line = export['lineInfo'][lineIdx]
    kr = line['kr']
    iel = line['iel']
    halfWidth = line['halfWidth']
    jIdx = data.jrad[kr] - 1
    iIdx = data.irad[kr] - 1

    lineCentre = data.alamb[kr]
    muIdx = 4

    # mu = 4 # for straight on
    # mu = 0 # for limb
    trimmedLines = []
    for m in range(len(data.zmu)):
        wl, intens = line_intensity_with_cont(data, kr, m)
        lowIdx = np.searchsorted(wl, lineCentre - halfWidth, side='right')-1
        highIdx = np.searchsorted(wl, lineCentre + halfWidth, side='left')+1
        trimmedLines.append(intens[:,lowIdx:highIdx])

    # These statements assume that we will go through the lineIdx in ascending
    # order, but, at the end of the day, we will.
    if len(export['wavelength']) <= lineIdx:
        export['wavelength'].append(torch.from_numpy(wl[lowIdx:highIdx].copy()))
    if len(export['mu']) <= lineIdx:
        export['mu'].append(data.zmu[muIdx])
        

    staticTemp = np.zeros_like(staticAltitudeGrid).astype(np.float32)
    staticNe = np.zeros_like(staticAltitudeGrid).astype(np.float32)
    staticVel = np.zeros_like(staticAltitudeGrid).astype(np.float32)

    if lineIdx == 0:
        export['nTime'].append(data.tg1.shape[0])
        export['beamSpectralIndex'].append(data.beamSpectralIndex)
        export['totalBeamEnergy'].append(data.totalBeamEnergy)
        export['beamPulseType'].append(data.beamPlulseType)
        export['cutoffEnergy'].append(data.cutoffEnergy)

    for t in range(export['nTime'][-1]):
        export['line'][lineIdx].append(torch.from_numpy(trimmedLines[muIdx][t].copy()))

        z = data.z1[t, ::-1]

        interp_static = lambda param: interp1d(z, param, assume_sorted=True)(staticAltitudeGrid)

        upperLevel = data.n1[t, ::-1, jIdx, iel]
        staticUpperLevel = interp_static(upperLevel)

        lowerLevel = data.n1[t, ::-1, iIdx, iel]
        staticLowerLevel = interp_static(lowerLevel)
        export['upperLevelPop'][lineIdx].append(torch.from_numpy(staticUpperLevel.copy()).float())
        export['lowerLevelPop'][lineIdx].append(torch.from_numpy(staticLowerLevel.copy()).float())

        if lineIdx == 0:
            temp = data.tg1[t, ::-1]
            staticTemp = interp_static(temp)

            ne = data.ne1[t, ::-1]
            staticNe = interp_static(ne)

            vel = data.vz1[t, ::-1]
            staticVel = interp_static(vel)

            nhG = data.n1[t, ::-1, 0, 0]
            staticNhG = interp_static(nhG)

            nhi = data.n1[t, ::-1, :5, 0].sum(axis=1)
            staticNhi = interp_static(nhi)

            nhii = data.n1[t, ::-1, 5, 0]
            staticNhii = interp_static(nhii)

            export['temperature'].append(torch.from_numpy(staticTemp.copy()).float())
            export['ne'].append(torch.from_numpy(staticNe.copy()).float())
            export['vel'].append(torch.from_numpy(staticVel.copy()).float())
            export['nhGround'].append(torch.from_numpy(staticNhG.copy()).float())
            export['nhi'].append(torch.from_numpy(staticNhi.copy()).float())
            export['nhii'].append(torch.from_numpy(staticNhii.copy()).float())
    return export

export = {}
export['lineInfo'] = [{'line': 'H_alpha', 'kr': 4, 'iel': 0, 'halfWidth': 1.4}, {'line': 'CaII k', 'kr': 20, 'iel': 1, 'halfWidth': 1.0}]
export['line'] = [[] for l in range(len(export['lineInfo']))]
export['temperature'] = []
export['ne'] = []
export['nhGround'] = []
export['nhi'] = []
export['nhii'] = []
export['upperLevelPop'] = [[] for l in range(len(export['lineInfo']))]
export['lowerLevelPop'] = [[] for l in range(len(export['lineInfo']))]
export['vel'] = []
export['nTime'] = []
export['beamSpectralIndex'] = []
export['totalBeamEnergy'] = []
export['beamPulseType'] = []
export['cutoffEnergy'] = []
export['mu'] = []
export['wavelength'] = []
export['z'] = torch.from_numpy(staticAltitudeGrid.copy())

# for i, f in enumerate(tqdm(files)):
#     data = RadynCdfLoader.load_vars(f, tags)

#     for lineIdx in range(len(export['lineInfo'])):
#         export = line_dict(data, export, lineIdx)

def async_load_radyn(f, tags):
    return RadynCdfLoader.load_vars(f, tags)

# SpacePy's CDF loader doesn't seem like multiple threads at the same time and our Radyn object doesn't pickle nicely, 
# but we can still load the next while we process the current one
with ThreadPoolExecutor(max_workers=1) as executor:
    radynFiles = [executor.submit(async_load_radyn, f, tags) for f in files]

    for res in tqdm(as_completed(radynFiles)):
        data = res.result()
        for lineIdx in range(len(export['lineInfo'])):
            export = line_dict(data, export, lineIdx)

with open(outputFolder + 'DoublePicoGigaPickle%d.pickle' % totalNum, 'wb') as p:
    pickle.dump(export, p)

