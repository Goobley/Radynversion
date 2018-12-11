from Inn2 import RadynversionNet
import os
import numpy as np
from scipy.interpolate import interp1d
import torch
import matplotlib.pyplot as plt

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
            "Halpha" : y_round_trip[:,model.outSchema.Halpha],
            "Ca8542" : y_round_trip[:,model.outSchema.Ca8542],
            "ne" : x_out[:,model.inSchema.ne],
            "temperature" : x_out[:,model.inSchema.temperature],
            "vel" : vel,
            "Halpha_true" : yz[0,model.outSchema.Halpha],
            "Ca8542_true" : yz[0,model.outSchema.Ca8542]
        }

        return results

def inversion_plots(results,batch_size,z,ca_data,ha_data):
    '''
    A function to plot the results of the inversions.

    Parameters
    ----------
    results : dict
        The results from the inversions.
    batch_size : int
        The number of samples to take from the latent space.
    z : torch.Tensor
        The height profiles of the RADYN grid.
    ca_data : list
        A concatenated list of the calcium wavelengths and intensities.
    ha_data : list
        A concatenated list of the hydrogen wavelengths and intensities.
    '''

    a = max(1.0 / batch_size, 0.002)
    fig, ax = plt.subplots(2,2,figsize=(10,10))
    ax2 = ax[0,0].twinx()
    for i in range(batch_size):
        ax[0,0].plot(z.cpu().numpy(),results["ne"][i].cpu().numpy(),c="C3",alpha=a)
        ax2.plot(z.cpu().numpy(),results["temperature"][i].cpu().numpy(),c="C1",alpha=a)
        ax[0,1].plot(z.cpu().numpy(),results["vel"][i].cpu().numpy(),c="C2",alpha=a)
        ax[1,0].plot(ha_data[0],results["Halpha"][i].cpu().numpy(),c="k",alpha=a)
        ax[1,1].plot(ca_data[0],results["Ca8542"][i].cpu().numpy(),c="k",alpha=a)

    ax[1,0].plot(ha_data[0],results["Halpha_true"].cpu().numpy(),"--",color="C3")
    ax[1,0].set_title(r"H$\alpha$")
    ax[1,0].set_ylabel("Normalised Intensity")
    ax[1,0].set_xlabel(r"Wavelength $\AA{}$")
    ax[1,1].plot(ca_data[0],results["Ca8542_true"].cpu().numpy(),"--",color="C3")
    ax[1,1].set_title(r"Ca II 8542$\AA{}$")
    ax[1,1].set_ylabel("Normalised Intensity")
    ax[1,1].set_xlabel(r"Wavelength $\AA{}$")
    ax[0,0].set_ylabel(r"$n_{e}$ [$cm^{-3}$]",color="C3")
    ax[0,0].set_xlabel("Height [cm]")
    ax[0,0].set_ylim(8,15)
    ax2.set_ylabel("T [K]",color="C1")
    ax2.set_ylim(3,8)
    ax[0,1].set_ylabel("v [km/s]",color="C2")
    ax[0,1].set_xlabel("Height [cm]")

    fig.tight_layout()
    fig.canvas.draw()