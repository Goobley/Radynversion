from Inn2 import RadynversionNet
import os
import numpy as np
from scipy.interpolate import interp1d
import torch
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.colors import PowerNorm,LinearSegmentedColormap
import matplotlib.ticker

__all__ = ["create_model","obs_files","interp_to_radyn_grid","normalise","inversion","inversion_plots","integrated_intensity","intensity_ratio","doppler_vel","lambda_0","variance","wing_idxs","oom_formatter","delta_lambda","lambda_0_wing","interp_fine"]

def create_model(filename,dev):
    '''
    A function to load the model to perform inversions on unseen data. This function also loads the height profile and wavelength grids from RADYN.

    Paramters
    ---------
    filename : str
        The path to the checkpoint file.
    dev : torch.device
        The hardware device to pass the model onto.

    Returns
    -------
    model : RadynversionNet
        The model with the loaded trained weights ready to do testing.
    checkpoint["z"] : torch.Tensor
        The height profile from the RADYN grid.
    '''

    if os.path.isfile(filename):
        print("=> loading checkpoint '%s'" % filename)
        checkpoint = torch.load(filename,map_location=dev)
        model = RadynversionNet(inputs=checkpoint["inRepr"],outputs=checkpoint["outRepr"],minSize=384).to(dev)
        model.load_state_dict(checkpoint["state_dict"])
        print("=> loaded checkpoint '%s' (total number of epochs trained for %d)" % (filename,checkpoint["epoch"]))
        return model, checkpoint["z"]
    else:
        print("=> no checkpoint found at '%s'" % filename)

def obs_files(path):
    '''
    A function to return a list of the files of the observations.

    Parameters
    ----------
    path : str
        The path to the observations.

    Returns
    -------
     : list
        The list of the paths to all of the observation files.
    '''

    return sorted([path + f for f in os.listdir(path) if f.endswith(".fits") and not f.startswith(".")])

def interp_to_radyn_grid(intensity_vector,centre_wvl,hw,wvl_range):
    '''
    A function to linearly interpolate the observational line profiles to the number of wavelength points in the RADYN grid.

    Parameters
    ----------
    intensity_vector : numpy.ndarray
        The intensity vector from a pixel in the CRISP image.
    centre_wvl : float
        The central measured wavelength obtained from the TWAVE1 keyword in the observartion's FITS header.
    hw : float
        The half-width of the line on the RADYN grid.
    wvl_range : numpy.ndarray
        The wavelength range from the observations.

    Returns
    -------
     : list
        A list of the interpolated wavelengths and intensities. Each element of the list is a numpy.ndarray.
    '''

    wvl_vector = np.linspace(centre_wvl-hw,centre_wvl+hw,num=30)
    interp = interp1d(wvl_range,intensity_vector,kind="linear")

    return [wvl_vector,interp(wvl_vector)]

def normalise(new_ca,new_ha):
    '''
    A function to normalise the spectral line profiles as the RADYN grid works on normalised profiles.

    Parameters
    ----------
    new_ca : numpy.ndarray
        The new calcium line interpolated onto the RADYN grid.
    new_ha : numpy.ndarray
        The new hydrogen line interpolated onto the RADYN grid.

    Returns
    -------
    new_ca : numpy.ndarray
        The interpolated calcium line normalised.
    new_ha : numpy.ndarray
        The interpolated hydrogen line normalised.
    '''

    peak_emission = max(np.amax(new_ca[1]),np.amax(new_ha[1]))

    new_ca[1] /= peak_emission
    new_ha[1] /= peak_emission

    return new_ca, new_ha

def inverse_velocity_conversion(out_velocities):
    '''
    A function to convert the calculated inverse velocities from the smooth space to the actual space.

    Parameters
    ----------
    out_velocities : torch.Tensor
        The velocity profiles obtained from the inversion.

    Returns
    -------
     : torch.Tensor
        The velocity profiles converted back to the actual space.
    '''

    v_sign = out_velocities / torch.abs(out_velocities)
    v_sign[torch.isnan(v_sign)] = 0

    return v_sign * (10**torch.abs(out_velocities) - 1.0)

def inversion(model,dev,ca_data,ha_data,batch_size):
    '''
    A function which performs the inversions on the spectral line profiles.

    Parameters
    ----------
    model : RadynversionNet
        The trained inversion model.
    dev : torch.device
        The hardware device to pass the model onto.
    ca_data : list
        A concatenated list of the calcium wavelengths and intensities.
    ha_data : list
        A concatenated list of the hydrogen wavelengths and intensities.
    batch_size : int
        The number of samples to take from the latent space.

    Returns
    -------
    results : dict
        The results of the inversions and the roundtrips on the line profiles.
    '''

    model.eval()
    with torch.no_grad():
        y = torch.ones((batch_size,2,ca_data[0].shape[0])) #sets up the input to the model by creating an array containing the line profiles a batch_size number of times such that the latent space can be sampled that many times for adequate confidence in the inversion
        y[:,0] *= torch.from_numpy(ha_data[1]).float()
        y[:,1] *= torch.from_numpy(ca_data[1]).float() #loads in the line profiles batch_size amount of times to be used with different samples drawn from the latent space
        yz = model.outSchema.fill({
            "Halpha" : y[:,0],
            "Ca8542" : y[:,1],
            "LatentSpace" : torch.randn
        }) #constructs the [y,z] pairs for the network
        x_out = model(yz.to(dev),rev=True)
        y_round_trip = model(x_out) #uses the calculated atmospheric parameters to generate the line profiles to see if they're the same
        vel = inverse_velocity_conversion(x_out[:,model.inSchema.vel])

        results = {
            "Halpha" : y_round_trip[:,model.outSchema.Halpha].cpu().numpy(),
            "Ca8542" : y_round_trip[:,model.outSchema.Ca8542].cpu().numpy(),
            "ne" : x_out[:,model.inSchema.ne].cpu().numpy(),
            "temperature" : x_out[:,model.inSchema.temperature].cpu().numpy(),
            "vel" : vel.cpu().numpy(),
            "Halpha_true" : yz[0,model.outSchema.Halpha].cpu().numpy(),
            "Ca8542_true" : yz[0,model.outSchema.Ca8542].cpu().numpy()
        }

        return results

def inversion_plots(results,z,ca_data,ha_data):
    '''
    A function to plot the results of the inversions.

    Parameters
    ----------
    results : dict
        The results from the inversions.m the latent space.
    z : torch.Tensor
        The height profiles of the RADYN grid.    
    ca_data : list
        A concatenated list of the calcium wavelengths and intensities.
    ha_data : list
        A concatenated list of the hydrogen wavelengths and intensities.
    '''

    fig, ax = plt.subplots(nrows=2,ncols=2,figsize=(9,7),constrained_layout=True)
    ax2 = ax[0,0].twinx()
    ca_wvls = ca_data[0]
    ha_wvls = ha_data[0]
    z_local = z / 1e8

    z_edges = [z_local[0] - 0.5*(z_local[1]-z_local[0])]
    for i in range(z_local.shape[0]-1):
        z_edges.append(0.5*(z_local[i]+z_local[i+1]))
    z_edges.append(z_local[-1] + 0.5*(z_local[-1]-z_local[-2]))
    z_edges = [float(f) for f in z_edges]
    ca_edges = [ca_wvls[0] - 0.5*(ca_wvls[1]-ca_wvls[0])]
    for i in range(ca_wvls.shape[0]-1):
        ca_edges.append(0.5*(ca_wvls[i]+ca_wvls[i+1]))
    ca_edges.append(ca_wvls[-1] + 0.5*(ca_wvls[-1]-ca_wvls[-2]))
    ha_edges = [ha_wvls[0] - 0.5*(ha_wvls[1]-ha_wvls[0])]
    for i in range(ha_wvls.shape[0]-1):
        ha_edges.append(0.5*(ha_wvls[i]+ha_wvls[i+1]))
    ha_edges.append(ha_wvls[-1] + 0.5*(ha_wvls[-1]-ha_wvls[-2]))
    ne_edges = np.linspace(8,15,num=101)
    temp_edges = np.linspace(3,8,num=101)
    vel_max = 2*np.max(np.median(results["vel"],axis=0))
    vel_min = np.min(np.median(results["vel"],axis=0))
    vel_min = np.sign(vel_min)*np.abs(vel_min)*2
    vel_edges = np.linspace(vel_min,vel_max,num=101)
    ca_max = 1.1*np.max(np.max(results["Ca8542"],axis=0))
    ca_min = 0.9*np.min(np.min(results["Ca8542"],axis=0))
    ca_edges_int = np.linspace(ca_min,ca_max,num=101)
    ha_max = 1.1*np.max(np.max(results["Halpha"],axis=0))
    ha_min = 0.9*np.min(np.min(results["Halpha"],axis=0))
    ha_edges_int = np.linspace(ha_min,ha_max,num=201)


    cmap_ne = [(51/255,187/255,238/255,0.0), (51/255, 187/255, 238/255, 1.0)]
    colors_ne = LinearSegmentedColormap.from_list('ne', cmap_ne)
    cmap_temp = [(238/255,119/255,51/255,0.0),(238/255,119/255,51/255,1.0)]
    colors_temp = LinearSegmentedColormap.from_list("temp",cmap_temp)
    cmap_vel = [(238/255,51/255,119/255,0.0),(238/255,51/255,119/255,1.0)]
    cmap_vel = LinearSegmentedColormap.from_list("vel",cmap_vel)

    ax[0,0].hist2d(torch.cat([z_local]*results["ne"].shape[0]).cpu().numpy(),results["ne"].reshape((-1,)),bins=(z_edges,ne_edges),cmap=colors_ne,norm=PowerNorm(0.3))
    ax[0,0].plot(z_local.cpu().numpy(),np.median(results["ne"],axis=0), "--",c="k")
    ax[0,0].set_ylabel(r"log $n_{e}$ [cm$^{-3}$]",color=(51/255,187/255,238/255))
    ax[0,0].set_xlabel("z [Mm]")
#     ax[0,0].xaxis.set_major_formatter(oom_formatter(8))
    ax2.hist2d(torch.cat([z_local]*results["temperature"].shape[0]).cpu().numpy(),results["temperature"].reshape((-1,)),bins=(z_edges,temp_edges),cmap=colors_temp,norm=PowerNorm(0.3))
    ax2.plot(z_local.cpu().numpy(),np.median(results["temperature"],axis=0),"--",c="k")
    ax2.set_ylabel("log T [K]",color=(238/255,119/255,51/255))
    ax[0,1].hist2d(torch.cat([z_local]*results["vel"].shape[0]).cpu().numpy(),results["vel"].reshape((-1,)),bins=(z_edges,vel_edges),cmap=cmap_vel,norm=PowerNorm(0.3))
    ax[0,1].plot(z_local.cpu().numpy(),np.median(results["vel"],axis=0),"--",c="k")
    ax[0,1].set_ylabel(r"v [kms$^{-1}$]",color=(238/255,51/255,119/255))
    ax[0,1].set_xlabel("z [Mm]")
#     ax[0,1].xaxis.set_major_formatter(oom_formatter(8))
    ax[1,0].plot(ha_data[0],results["Halpha_true"],"--")
    ax[1,0].hist2d(np.concatenate([ha_wvls]*results["Halpha"].shape[0]),results["Halpha"].reshape((-1,)),bins=(ha_edges,ha_edges_int),cmap="gray_r",norm=PowerNorm(0.3))
    ax[1,0].set_title(r"H$\alpha$")
    ax[1,0].set_ylabel("Normalised Intensity")
    ax[1,0].set_xlabel(r"Wavelength [$\AA{}$]")
    ax[1,0].xaxis.set_major_locator(plt.MaxNLocator(5))
    ax[1,1].hist2d(np.concatenate([ca_wvls]*results["Ca8542"].shape[0]),results["Ca8542"].reshape((-1,)),bins=(ca_edges,ca_edges_int),cmap="gray_r",norm=PowerNorm(0.3))
    ax[1,1].set_title(r"Ca II 8542$\AA{}$")
    ax[1,1].plot(ca_data[0],results["Ca8542_true"],"--")
    ax[1,1].set_xlabel(r"Wavelength [$\AA{}$]")
    ax[1,1].xaxis.set_major_locator(plt.MaxNLocator(5))

class oom_formatter(matplotlib.ticker.ScalarFormatter):
    '''
    Matplotlib formatter for changing the number of orders of magnitude displayed on an axis as well as the number of decimal points.

    Adapted from: https://stackoverflow.com/questions/42656139/set-scientific-notation-with-fixed-exponent-and-significant-digits-for-multiple
    '''

    def __init__(self,order=0,fformat="%1.1f",offset=True,math_text=True):
        self.oom = order
        self.fformat = fformat
        matplotlib.ticker.ScalarFormatter.__init__(self,useOffset=offset,useMathText=math_text)
        
    def _set_orderOfMagnitude(self,nothing):
        self.orderOfMagnitude = self.oom
        
    def _set_format(self, v_min, v_max):
        self.format = self.fformat
        if self._useMathText:
            self.format = "$%s$" % matplotlib.ticker._mathdefault(self.format)

def integrated_intensity(idx_range,intensity_vector):
    '''
    A function to find the integrated intensity over a wavelength range of a spectral line.

    Parameters
    ----------
    idx_range : range
        The range of indices to integrate over.
    intensity_vector : numpy.ndarray
        The vector of spectral line intensities.
    '''

    total = 0
    for idx in idx_range:
        total += intensity_vector[idx]
    
    return total / len(idx_range)

def intensity_ratio(blue_intensity,red_intensity):
    '''
    A function that calculates the intensity ratio of two previously integrated intensities.
    '''

    return blue_intensity / red_intensity

def doppler_vel(l,delta_l):
    return (delta_l / l) * 3e5 #calculates the doppler velocity in km/s

def lambda_0(wvls,ints):
    '''
    Calculates the intensity-averaged line core.
    '''

    num = np.sum(np.multiply(ints,wvls))
    den = np.sum(ints)

    return num / den

def variance(wvls,ints,l_0):
    '''
    Calculates the variance of the spectral line w.r.t. the intensity-averaged line core.
    '''

    num = np.sum(np.multiply(ints,(wvls-l_0)**2))
    den = np.sum(ints)

    return num / den

def wing_idxs(wvls,ints,var,l_0):
    '''
    A function to work out the index range for the wings of a spectral line. This is working on the definition of wings that says the wings are defined as being one standard deviation away from the intensity-averaged line core.
    '''

    blue_wing_start = 0 #blue wing starts at the shortest wavelength
    red_wing_end = wvls.shape[0] - 1 #red wing ends at the longest wavelength

    blue_end_wvl = l_0 - np.sqrt(var)
    red_start_wvl = l_0 + np.sqrt(var)

    blue_wing_end = np.argmin(np.abs(wvls - blue_end_wvl))
    red_wing_start = np.argmin(np.abs(wvls - red_start_wvl))

    return range(blue_wing_start,blue_wing_end+1), range(red_wing_start,red_wing_end+1)

def delta_lambda(wing_idxs,wvls):
    '''
    Calculates the half-width wavelength of an intensity range.

    Parameters
    ----------
    wing_idxs : range
        The range of the indices of the intensity region in question.
    wvls : numpy.ndarray
        The wavelengths corresponding to the intensity region in question.
    '''

    return len(wing_idxs)*(wvls[1] - wvls[0])/2

def lambda_0_wing(wing_idxs,wvls,delta_lambda):
    '''
    Calculates the central wavelength of an intensity range.

    Parameters
    ----------
    wing_idxs : range
        The range of the indices of the intensity region in question.
    wvls : numpy.ndarray
        The wavelengths corresponding to the intensity region in question.
    delta_lambda : float
        The half-width wavelength of an intensity range.
    '''

    return wvls[list(wing_idxs)[-1]] - delta_lambda

def interp_fine(spec_line):
    '''
    Interpolates the spectral line onto a finer grid for more accurate calculations for the wing properties.
    '''

    x, y = spec_line
    x_new = np.linspace(x[0],x[-1],num=1001)
    y_new = interp1d(x,y)(x_new)

    return np.array([x_new,y_new])