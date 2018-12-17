from Inn2 import RadynversionNet
import os
import numpy as np
from scipy.interpolate import interp1d
import torch
import matplotlib.pyplot as plt
from skimage.draw import line_aa
import matplotlib as mpl
from matplotlib.colors import PowerNorm,LinearSegmentedColormap

__all__ = ["create_model","obs_files","interp_to_radyn_grid","normalise","inversion","inversion_plots"]

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
    checkpoint["wls"][0] : torch.Tensor
        The H alpha wavelength points from the RADYN grid.
    checkpoint["wls][1] : torch.Tensor
        The Ca II 8542 wavelength points from the RADYN grid.
    '''

    if os.path.isfile(filename):
        print("=> loading checkpoint '%s'" % filename)
        checkpoint = torch.load(filename,map_location=dev)
        model = RadynversionNet(inputs=checkpoint["inRepr"],outputs=checkpoint["outRepr"],minSize=384).to(dev)
        model.load_state_dict(checkpoint["state_dict"])
        print("=> loaded checkpoint '%s' (total number of epochs trained for %d)" % (filename,checkpoint["epoch"]))
        return model, checkpoint["z"], checkpoint["wls"][0].cpu().numpy(), checkpoint["wls"][1].cpu().numpy()
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

def inversion_plots_acc(results,z,ca_data,ha_data):
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

    fig, ax = plt.subplots(nrows=2,ncols=2,figsize=(9,7))
    ax2 = ax[0,0].twinx()
    ca_wvls = ca_data[0]
    ha_wvls = ha_data[0]

    z_edges = [z[0] - 0.5*(z[1]-z[0])]
    for i in range(z.shape[0]-1):
        z_edges.append(0.5*(z[i]+z[i+1]))
    z_edges.append(z[-1] + 0.5*(z[-1]-z[-2]))
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

    ax[0,0].hist2d(torch.cat([z]*results["ne"].shape[0]).cpu().numpy(),results["ne"].reshape((-1,)),bins=(z_edges,ne_edges),cmap=colors_ne,norm=PowerNorm(0.3))
    ax[0,0].plot(z.cpu().numpy(),np.median(results["ne"],axis=0), "--",c="k")
    ax[0,0].set_ylabel(r"$n_{e}$ [cm$^{-3}$]",color=(51/255,187/255,238/255))
    ax[0,0].set_xlabel("z [cm]")
    ax2.hist2d(torch.cat([z]*results["temperature"].shape[0]).cpu().numpy(),results["temperature"].reshape((-1,)),bins=(z_edges,temp_edges),cmap=colors_temp,norm=PowerNorm(0.3))
    ax2.plot(z.cpu().numpy(),np.median(results["temperature"],axis=0),"--",c="k")
    ax2.set_ylabel("T [K]",color=(238/255,119/255,51/255))
    ax[0,1].hist2d(torch.cat([z]*results["vel"].shape[0]).cpu().numpy(),results["vel"].reshape((-1,)),bins=(z_edges,vel_edges),cmap=cmap_vel,norm=PowerNorm(0.3))
    ax[0,1].plot(z.cpu().numpy(),np.median(results["vel"],axis=0),"--",c="k")
    ax[0,1].set_ylabel(r"v [kms$^{-1}$]",color=(238/255,51/255,119/255))
    ax[0,1].set_xlabel("z [cm]")
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

    fig.tight_layout()

def integrated_intensity(idx_range,intensity_vector):
    total = 0
    for idx in idx_range:
        total += intensity_vector[idx]
    
return total / len(idx_range)

def intensity_ratio(blue_intensity,red_intensity):
    return blue_intensity / red_intensity