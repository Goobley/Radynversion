
# coding: utf-8

# ## Radynversion Training Notebook
# 
# This notebook is used to train the Radynversion neural network. It requires the following packages:
# 
# `numpy`
# 
# `scipy`
# 
# `matplotlib`
# 
# `FrEIA: https://github.com/VLL-HD/FrEIA`
# 
# `pytorch >= 0.4.1` (only tested on `0.4.1` but will probably be updated to `1.0.x` soon -- I don't forsee any problems with this).
# 
# An NVIDIA GPU with CUDA and > 2GB VRAM is strongly recommended if you are going to attempt to train a Radynversion model. With a 1050 Ti, the full 12000 epochs are trained in under a day.
# 
# The hyperparameters listed here (learning rate, loss weights etc.) have all been empirically found to work, but changing the data may well necessitate changing these.
# 
# To (re)train Radynversion this notebook can be run pretty much from top to bottom, with only a little tweaking of the hyperparameters necessary if you change the the complexity of the input data.
# 
# A lot of the heavy lifting functions are in the files `Inn2.py` and `Loss.py`.
# 
# Please forgive the massive blobs of plotting code, the same technique is used to plot the results from the inversions and is nicely tucked away in `utils.py`, most of that code organically grew in this notebook!
# 
# To (re)train the model you will also need the training data. Either look at the ridiculously named `ExportSimpleLineBlobForTraining.py` to export the required data from your own RADYN sims/move around the atmospheric nodes etc. or use our _even_ more ridiculously named training data `DoublePicoGigaPickle50.pickle` which will be made available, along with the trained for the initial release of Radynversion on Radynversion's Github releases page. The training pickle contains all of the snapshots from the Fokker-Planck RADYN simulations in the F-CHROMA grid, sampled at the 50 atmospheric points detailed in Osborne, Armstrong, and Fletcher (2019).

# In[1]:


#get_ipython().magic('matplotlib notebook')

from Inn2 import RadynversionNet, AtmosData, RadynversionTrainer
import loss as Loss
import pickle
import numpy as np
import scipy
from scipy.stats import multivariate_normal as mvn
from scipy.special import logit, expit
from scipy.stats import uniform, gaussian_kde, ks_2samp, anderson_ksamp
from scipy import stats
from scipy.signal import butter, lfilter, freqs, resample
from scipy.interpolate import interp1d
#import matplotlib
#matplotlib.use('Agg')
import matplotlib.pyplot as plt
#import matplotlib.pyplot as plt
import torch
import torch.optim
import torch.utils.data
import h5py
import os
from sys import exit

from time import time

from FrEIA.framework import InputNode, OutputNode, Node, ReversibleGraphNet
from FrEIA.modules import rev_multiplicative_layer, F_fully_connected, F_conv

import data_maker as data_maker

dev = 'cuda' if torch.cuda.is_available() else 'cpu'
#from IPython.core.display import display, HTML
#display(HTML("<style>.container { width:95% !important; }</style>"))

# global parameters
sig_model = 'sg'   # the signal model to use
run_label='gpu7'
out_dir = "/home/hunter.gabbard/public_html/CBC/cINNamon/gausian_results/multipar/%s/" % run_label
do_posterior_plots=True
ndata=64           # length of 1 data sample
ndim_x=3           # number of parameters to PE on
ndim_y = ndata     # length of 1 data sample
ndim_z = 3       # size of latent space
Ngrid=64
n_neurons = 0
ndim_tot = max(ndim_x,ndim_y+ndim_z) + n_neurons # 384     
r = 3              # the grid dimension for the output tests
sigma = 0.2        # the noise std
seed = 1           # seed for generating data
test_split = r*r   # number of testing samples to use
N_samp = 8000 # number of test samples to use after training
plot_cadence = 10  # make plots every N iterations
numInvLayers=5
dropout=0.0
filtsize = 3       # TODO
clamp=2.0          # TODO
tot_dataset_size=int(1e6) # TODO really should use 1e8 once cpu is fixed
tot_epoch=11000
lr=1.0e-3
zerosNoiseScale=5e-2
wPred=300.0        #4000.0
wLatent= 300.0     #900.0
wRev= 400.0        #1000.0
latentAlphas=None #[8,11]
backwardAlphas=None # [1.4, 2, 5.5, 7]
conv_nn = False    # Choose to use convolutional layers. TODO
multi_par=True
load_dataset=True
do_contours=True
do_mcmc=True
dataLocation1 = 'benchmark_data.h5py'
T = 1.0           # length of time series (s)
dt = T/ndata        # sampling time (Sec)
fnyq = 0.5/dt   # Nyquist frequency (Hz)
if multi_par==True: bound = [0.0,1.0,0.0,1.0,0.0,1.0*fnyq,0.0,3.0,0.0,1.0]
else: bound = [0.0,1.0,0.0,1.0] # effective bound for the liklihood

def make_contour_plot(ax,x,y,dataset,color='red',flip=False, kernel_lalinf=False, kernel_cnn=False, bounds=[0.0,1.0,0.0,1.0]):
    """ Module used to make contour plots in pe scatter plots.
    Parameters
    ----------
    ax: matplotlib figure
        a matplotlib figure instance
    x: 1D numpy array
        pe sample parameters for x-axis
    y: 1D numpy array
        pe sample parameters for y-axis
    dataset: 2D numpy array
        array containing both parameter estimates
    color:
        color of contours in plot
    flip:
        if True: transpose parameter estimates array. if False: do not transpose parameter estimates
        TODO: This is not used, so should remove
    Returns
    -------
    kernel: scipy kernel
        gaussian kde of the input dataset
    """
    # Make a 2d normed histogram
    H,xedges,yedges=np.histogram2d(x,y,bins=10,normed=True)

    if flip == True:
        H,xedges,yedges=np.histogram2d(y,x,bins=10,normed=True)
        dataset = np.array([dataset[1,:],dataset[0,:]])

    norm=H.sum() # Find the norm of the sum
    # Set contour levels
    contour1=0.99
    contour2=0.90
    contour3=0.68

    # Set target levels as percentage of norm
    target1 = norm*contour1
    target2 = norm*contour2
    target3 = norm*contour3

    # Take histogram bin membership as proportional to Likelihood
    # This is true when data comes from a Markovian process
    def objective(limit, target):
        w = np.where(H>limit)
        count = H[w]
        return count.sum() - target

    # Find levels by summing histogram to objective
    level1= scipy.optimize.bisect(objective, H.min(), H.max(), args=(target1,))
    level2= scipy.optimize.bisect(objective, H.min(), H.max(), args=(target2,))
    level3= scipy.optimize.bisect(objective, H.min(), H.max(), args=(target3,))

    # For nice contour shading with seaborn, define top level
    #level4=H.max()
    levels=[level1,level2,level3]

    # Pass levels to normed kde plot
    #sns.kdeplot(x,y,shade=True,ax=ax,n_levels=levels,cmap=color,alpha=0.5,normed=True)
    #X, Y = np.mgrid[np.min(x):np.max(x):100j, np.min(y):np.max(y):100j]
    X, Y = np.mgrid[bounds[0]:bounds[1]:100j, bounds[2]:bounds[3]:100j]
    positions = np.vstack([X.ravel(), Y.ravel()])
    if not kernel_lalinf or not kernel_cnn: kernel = gaussian_kde(dataset)
    Z = np.reshape(kernel(positions).T, X.shape)
    ax.contour(X,Y,Z,levels=levels,alpha=0.5,colors=color,linewidths=0.6)
    #ax.set_aspect('equal')

    return kernel

def overlap_tests(pred_samp,lalinf_samp,true_vals,kernel_cnn,kernel_lalinf):
    """ Perform Anderson-Darling, K-S, and overlap tests
    to get quantifiable values for accuracy of GAN
    PE method
    Parameters
    ----------
    pred_samp: numpy array
        predicted PE samples from CNN
    lalinf_samp: numpy array
        predicted PE samples from lalinference
    true_vals:
        true scalar point values for parameters to be estimated (taken from GW event paper)
    kernel_cnn: scipy kde instance
        gaussian kde of CNN results
    kernel_lalinf: scipy kde instance
        gaussian kde of lalinference results
    Returns
    -------
    ks_score:
        k-s test score
    ad_score:
        anderson-darling score
    beta_score:
        overlap score. used to determine goodness of CNN PE estimates
    """

    # do k-s test
    ks_mc_score = ks_2samp(pred_samp[:,0].reshape(pred_samp[:,0].shape[0],),lalinf_samp[0][:])
    ks_q_score = ks_2samp(pred_samp[:,1].reshape(pred_samp[:,1].shape[0],),lalinf_samp[1][:])
    ks_score = np.array([ks_mc_score,ks_q_score])

    # do anderson-darling test
    ad_mc_score = anderson_ksamp([pred_samp[:,0].reshape(pred_samp[:,0].shape[0],),lalinf_samp[0][:]])
    ad_q_score = anderson_ksamp([pred_samp[:,1].reshape(pred_samp[:,1].shape[0],),lalinf_samp[1][:]])
    ad_score = [ad_mc_score,ad_q_score]

    # compute overlap statistic
    comb_mc = np.concatenate((pred_samp[:,0].reshape(pred_samp[:,0].shape[0],1),lalinf_samp[0][:].reshape(lalinf_samp[0][:].shape[0],1)))
    comb_q = np.concatenate((pred_samp[:,1].reshape(pred_samp[:,1].shape[0],1),lalinf_samp[1][:].reshape(lalinf_samp[1][:].shape[0],1)))
    X, Y = np.mgrid[np.min(comb_mc):np.max(comb_mc):100j, np.min(comb_q):np.max(comb_q):100j]
    positions = np.vstack([X.ravel(), Y.ravel()])
    #cnn_pdf = np.reshape(kernel_cnn(positions).T, X.shape)
    #print(positions.shape,pred_samp.shape)
    cnn_pdf = kernel_cnn.pdf(positions)

    #X, Y = np.mgrid[np.min(lalinf_samp[0][:]):np.max(lalinf_samp[0][:]):100j, np.min(lalinf_samp[1][:]):np.max(lalinf_samp[1][:]):100j]
    positions = np.vstack([X.ravel(), Y.ravel()])
    #lalinf_pdf = np.reshape(kernel_lalinf(positions).T, X.shape)
    lalinf_pdf = kernel_lalinf.pdf(positions)

    beta_score = np.divide(np.sum( cnn_pdf*lalinf_pdf ),
                              np.sqrt(np.sum( cnn_pdf**2 ) *
                              np.sum( lalinf_pdf**2 )))


    return ks_score, ad_score, beta_score

# Load the training data -- you will need to modify this path.

# In[4]:

def store_pars(f,pars):
    for i in pars.keys():
        f.write("%s: %s\n" % (i,str(pars[i])))
    f.close()

# store hyperparameters for posterity
f=open("%s_run-pars.txt" % run_label,"w+")
pars_to_store={"sigma":sigma,"ndata":ndata,"T":T,"seed":seed,"n_neurons":n_neurons,"bound":bound,"conv_nn":conv_nn,"filtsize":filtsize,"dropout":dropout,
               "clamp":clamp,"ndim_z":ndim_z,"tot_epoch":tot_epoch,"lr":lr, "latentAlphas":latentAlphas, "backwardAlphas":backwardAlphas,
               "zerosNoiseScale":zerosNoiseScale,"wPred":wPred,"wLatent":wLatent,"wRev":wRev,"tot_dataset_size":tot_dataset_size,
               "numInvLayers":numInvLayers}
store_pars(f,pars_to_store)

# generate data
if not load_dataset:
    pos, labels, x, sig = data_maker.generate(
        model=sig_model,
        tot_dataset_size=tot_dataset_size, # 1e6
        ndata=ndata,
        sigma=sigma,
        prior_bound=bound,
        seed=seed,
        multi_par=multi_par,
        dt=dt,
        fnyq=fnyq
    )
    hf = h5py.File('benchmark_data.h5py', 'w')
    hf.create_dataset('pos', data=pos)
    hf.create_dataset('labels', data=labels)
    hf.create_dataset('x', data=x)
    hf.create_dataset('sig', data=sig)

    print('Samples generated')

if multi_par==True: true_post = np.zeros((r,r,N_samp,ndim_x))
else: true_post = np.zeros((r,r,N_samp,2))

data = AtmosData([dataLocation1], test_split, resampleWl=None)
data.split_data_and_init_loaders(500)

# Construct the DataSchemas (descriptions of the network's inputs and outputs), and construct the network model using these.

# In[6]:

# seperate the test data for plotting
pos_test = data.pos_test
labels_test = data.labels_test
sig_test = data.sig_test

lik = np.zeros((r,r,Ngrid*Ngrid))

print('Computing MCMC posterior samples')
if do_mcmc or not load_dataset:
    if multi_par==True:
        r_mcmc=1
        # choose waveform in test set to do mcmc on
        cnt=3
    for i in range(r_mcmc):
        for j in range(r_mcmc):
            mvec,cvec,temp,post_points = data_maker.get_lik(np.array(labels_test[cnt,:]).flatten(),
                                                      np.array(pos_test[cnt,:]),n_grid=Ngrid,sig_model=sig_model,
                                                      sigma=sigma,xvec=data.x,bound=bound,multi_par=multi_par,
                                                      N_samp=N_samp)
            lik[i,j,:] = temp.flatten()
            idx = np.random.randint(0,post_points.shape[0],N_samp)
            true_post[i,j,:] = post_points[idx]
            cnt += 1

    # save computationaly expensive mcmc/waveform runs
    if load_dataset==True:
        hf = h5py.File('benchmark_data.h5py', 'w')
        hf.create_dataset('pos', data=data.pos)
        hf.create_dataset('labels', data=data.labels)
        hf.create_dataset('x', data=data.x)
        hf.create_dataset('sig', data=data.sig)
    hf.create_dataset('post_points', data=post_points)
    hf.create_dataset('lik', data=lik)
    hf.create_dataset('true_post', data=true_post)
    hf.close()

else:
    lik=h5py.File(dataLocation1, 'r')['lik'][:]
    true_post=h5py.File(dataLocation1, 'r')['true_post'][:]
    #post_points=h5py.File(dataLocation1, 'r')['post_points'][:]

# plot the test data examples
#plt.figure(figsize=(3,3))
fig_post, axes = plt.subplots(nrows=int(r),ncols=int(r))
cnt = 0
for i in range(r):
    for j in range(r):
        axes[i,j].plot(data.x,np.array(labels_test[cnt,:]),'.')
        axes[i,j].plot(data.x,np.array(sig_test[cnt,:]),'-')
        cnt += 1
        axes[i,j].axis([0,1,-1.5,1.5])
plt.savefig("%stest_distribution.pdf" % out_dir,dpi=360)
plt.close()

# setup output directory - if it does not exist
os.system('mkdir -p %s' % out_dir)

#inRepr = [('ne', data.ne.shape[1]), ('temperature', data.temperature.shape[1]), ('vel', data.vel.shape[1]), ('!!PAD',)]
#outRepr = [('LatentSpace', 200), ('!!PAD',), ('Halpha', data.lines[0].shape[1]), ('Ca8542', data.lines[1].shape[1])]
inRepr = [('amp', 1), ('t0', 1), ('tau', 1), ('!!PAD',)]
outRepr = [('LatentSpace', ndim_z), ('!!PAD',), ('timeseries', data.atmosOut.shape[1])]
model = RadynversionNet(inRepr, outRepr, dropout=dropout, zeroPadding=0, minSize=ndim_tot, numInvLayers=numInvLayers)

# In[1]:


# Optionally print the constructed DataSchemas and the string representation of the model.
# model.inSchema, model.outSchema, model


# Construct the class that trains the model, the initial weighting between the losses, learning rate, and the initial number of epochs to train for.

# In[8]:


trainer = RadynversionTrainer(model, data, dev)
trainer.training_params(tot_epoch, lr=lr, zerosNoiseScale=zerosNoiseScale, wPred=wPred, wLatent=wLatent, wRev=wRev,
                        loss_latent=Loss.mmd_multiscale_on(dev, alphas=latentAlphas),
                        loss_backward=Loss.mmd_multiscale_on(dev, alphas=backwardAlphas),
                        loss_fit=Loss.mse)
totalEpochs = 0

# Train the model for these first epochs with a nice graph that updates during training.

# In[ ]:


losses = []
beta_score_hist=[]
lossVec = [[] for _ in range(4)]
lossLabels = ['L2 Line', 'MMD Latent', 'MMD Reverse', 'L2 Reverse']
out = None
alphaRange, mmdF, mmdB, idxF, idxB = [1,1], [1,1], [1,1], 0, 0
try:
    tStart = time()
    for epoch in range(trainer.numEpochs):
        print('Epoch %s/%s' % (str(epoch),str(trainer.numEpochs)))
        totalEpochs += 1

        trainer.scheduler.step()
        
        loss, indLosses = trainer.train(epoch)
        
        if ((epoch % 1 == 0) & (epoch>5)):
            fig, axis = plt.subplots(4,1, figsize=(10,8))
            #fig.show()
            fig.canvas.draw()
            axis[0].clear()
            axis[1].clear()
            axis[2].clear()
            axis[3].clear()
            for i in range(len(indLosses)):
                lossVec[i].append(indLosses[i])
            losses.append(loss)
            fig.suptitle('Current Loss: %.2e, min loss: %.2e' % (loss, np.nanmin(np.abs(losses))))
            axis[0].semilogy(np.arange(len(losses)), np.abs(losses))
            for i, lo in enumerate(lossVec):
                axis[1].semilogy(np.arange(len(losses)), lo, '--', label=lossLabels[i])
            axis[1].legend(loc='upper left')
            tNow = time()
            elapsed = int(tNow - tStart)
            eta = int((tNow - tStart) / (epoch + 1) * trainer.numEpochs) - elapsed

            if epoch % 2 == 0:
                mses = trainer.test(maxBatches=1)
                lineProfiles = mses[2]
                
            if epoch % 10 == 0:
                alphaRange, mmdF, mmdB, idxF, idxB = trainer.review_mmd()
                
            axis[3].semilogx(alphaRange, mmdF, label='Latent Space')
            axis[3].semilogx(alphaRange, mmdB, label='Backward')
            axis[3].semilogx(alphaRange[idxF], mmdF[idxF], 'ro')
            axis[3].semilogx(alphaRange[idxB], mmdB[idxB], 'ro')
            axis[3].legend()

            testTime = time() - tNow
            axis[2].plot(lineProfiles[0, model.outSchema.timeseries].cpu().numpy())
            for a in axis:
                a.grid()
            axis[3].set_xlabel('Epochs: %d, Elapsed: %d s, ETA: %d s (Testing: %d s)' % (epoch, elapsed, eta, testTime))
            
                
            fig.canvas.draw()
            fig.savefig('%slosses.pdf' % out_dir)

        if do_posterior_plots and ((epoch % plot_cadence == 0) & (epoch>5)):

            print('Making posterior plots ...')
            # choose which test sample
            cnt = 3
            # initialize plot for showing testing results
            fig_post, axes = plt.subplots(ndim_x,ndim_x,figsize=(10,8))

            # convert data into correct format
            y_samps = np.tile(np.array(labels_test[cnt,:]),N_samp).reshape(N_samp,ndim_y)
            y_samps = torch.tensor(y_samps, dtype=torch.float)
            #y_samps += y_noise_scale * torch.randn(N_samp, ndim_y)
            y_samps = torch.cat([torch.randn(N_samp, ndim_z), #zeros_noise_scale * 
                torch.zeros(N_samp, ndim_tot - ndim_y - ndim_z),
                y_samps], dim=1)
            y_samps = y_samps.to(dev)

            if conv_nn == True: y_samps = y_samps.reshape(y_samps.shape[0],y_samps.shape[1],1,1)
            rev_x = model(y_samps, rev=True)
            rev_x = rev_x.cpu().data.numpy()

            # make contour plots of INN time series predictions
            xvec = np.arange(ndata)/float(ndata)
            w=3.8
            p=0.5
            sig_preds=[]
            for i in range(len(rev_x)):
                sig_preds.append([rev_x[i,0]*np.sin(2.0*np.pi*w*fnyq*(x-rev_x[i,1]) + 2.0*np.pi*p)*np.exp(-((x-rev_x[i,1])/rev_x[i,2])**2) for x in xvec])
            sig_preds=np.array(sig_preds)

            # compute percentile curves
            perc_90 = []
            perc_75 = []
            perc_25 = []
            perc_5 = []
            for n in range(sig_preds.shape[1]):
                perc_90.append(np.percentile(sig_preds[:,n], 90))
                perc_75.append(np.percentile(sig_preds[:,n], 75))
                perc_25.append(np.percentile(sig_preds[:,n], 25))
                perc_5.append(np.percentile(sig_preds[:,n], 5))

            fig_tseries, ax_tseries = plt.subplots(1,1,figsize=(10,8))

            #for i in range(10):
            #    ax_tseries.plot(data.x*ndata,sig_preds[i],color='grey',alpha=0.5)
            ax_tseries.fill_between(np.linspace(0,len(perc_90),num=len(perc_90)),perc_90, perc_5, lw=0,facecolor='#d5d8dc')
            ax_tseries.fill_between(np.linspace(0,len(perc_75),num=len(perc_75)),perc_75, perc_25, lw=0,facecolor='#808b96')
            ax_tseries.plot(data.x*ndata,labels_test[cnt,:],color='black')
            ax_tseries.plot(data.x*ndata,np.array(sig_test[cnt,:]),'-',color='cyan')
            fig_post.savefig('%stseries_preds_%s.pdf' % (out_dir,epoch), dpi=360)
            fig_tseries.savefig('%stseries_preds_latest.pdf' % out_dir, dpi=360)

            # make corner plot for one test waveform
            cnt_plot = ndim_x
            cnt_y = 0
            cnt_beta = 0
            beta_score_loop = []
            #plot_labels = [r"$Amplitude$", r"$t0$", r"$f$", r"$\phi$", r"$\sigma$"]
            plot_labels = [r"$Amplitude$", r"$t0$", r"$\sigma$"]
            for i in range(ndim_x):
                for j in range(cnt_plot):
                    axes[-j+cnt_plot-1+cnt_y,i].clear()
                    #axes[-j+cnt_plot-1+cnt_y,i].contour(mvec,cvec,lik[0,0,:].reshape(Ngrid,Ngrid),levels=[0.68,0.9,0.99])
                    axes[-j+cnt_plot-1+cnt_y,i].scatter(rev_x[:,i], rev_x[:,-j+cnt_plot-1+cnt_y],edgecolors='none',alpha=1.0,s=0.8,color='red')
                    axes[-j+cnt_plot-1+cnt_y,i].scatter(true_post[0,0,:,i],true_post[0,0,:,-j+cnt_plot-1+cnt_y],edgecolors='none',s=0.8,alpha=0.5,color='blue') # s=0.3
                    axes[-j+cnt_plot-1+cnt_y,i].plot(pos_test[cnt,i],pos_test[cnt,-j+cnt_plot-1+cnt_y],'+c',markersize=5,markeredgewidth=0.5,alpha=0.75)
                    axes[-j+cnt_plot-1+cnt_y,i].set_xlabel(plot_labels[i])
                    axes[-j+cnt_plot-1+cnt_y,i].set_ylabel(plot_labels[-j+cnt_plot-1+cnt_y])
                    #axes[-j+cnt_plot-1+cnt_y,i].tick_params(labelsize=2)
                    #axes[-j+cnt_plot-1+cnt_y,i].axis([bound[i*2],bound[(i*2)+1],bound[(-j+cnt_plot-1+cnt_y)*2],bound[((-j+cnt_plot-1+cnt_y)*2)+1]])
                    axes[-j+cnt_plot-1+cnt_y,i].axis(bound[:4])

                    # set some axis to be off
                    if (-j+cnt_plot-1+cnt_y) != (ndim_x-1):
                        axes[-j+cnt_plot-1+cnt_y,i].get_xaxis().set_visible(False)
                    if (i) != 0:
                        axes[-j+cnt_plot-1+cnt_y,i].get_yaxis().set_visible(False)

                    try:
                        if do_contours:
                            if conv_nn==True: rev_x=rev_x.reshape(rev_x.shape[0],rev_x.shape[1])
                            contour_y = np.reshape(rev_x[:,-j+cnt_plot-1+cnt_y], (rev_x[:,-j+cnt_plot-1+cnt_y].shape[0]))
                            contour_x = np.reshape(rev_x[:,i], (rev_x[:,i].shape[0]))
                            contour_dataset = np.array([contour_x,contour_y])
                            kernel_cnn = make_contour_plot(axes[-j+cnt_plot-1+cnt_y,i],contour_x,contour_y,contour_dataset,'red',flip=False, kernel_cnn=False)

                            # run overlap tests on results
                            contour_x = np.reshape(true_post[0,0,:,-j+cnt_plot-1+cnt_y], (true_post[0,0,:,-j+cnt_plot-1+cnt_y].shape[0]))
                            contour_y = np.reshape(true_post[0,0,:,i], (true_post[0,0,:,i].shape[0]))
                            contour_dataset = np.array([contour_x,contour_y])
                            kernel_mcmc = make_contour_plot(axes[-j+cnt_plot-1+cnt_y,i],contour_x,contour_y,contour_dataset,'blue',flip=True, kernel_cnn=False) # gaussian_kde(contour_dataset)
                            ks_score, ad_score, beta_score = overlap_tests(np.vstack((rev_x[:,i],rev_x[:,-j+cnt_plot-1+cnt_y])),
                                                                                   np.vstack((true_post[0,0,:,i],true_post[0,0,:,-j+cnt_plot-1+cnt_y])),
                                                                                   np.vstack((pos_test[cnt,i],pos_test[cnt,-j+cnt_plot-1+cnt_y])),kernel_cnn,kernel_mcmc)
                            axes[-j+cnt_plot-1+cnt_y,i].legend(['Overlap: %s' % str(np.round(beta_score,3))])#, prop={'size': 3})

                            # save and plot history of overlap score
                            beta_score_loop.append([beta_score])
                            cnt_beta+=1
                            if cnt_beta==ndim_x: 
                                beta_score_hist.append(np.mean(beta_score_loop))
                                fig_beta, ax_beta = plt.subplots(1,1, figsize=(10,8))
                                ax_beta.plot(np.linspace(plot_cadence,epoch,len(beta_score_hist)),beta_score_hist)
                                fig_beta.savefig('%sbeta_history.pdf' % out_dir,dpi=360)
                    except Exception as e: 
                        print(e)
                    #    exit()
                    #except:
                        #pass
                cnt_plot -= 1
                cnt_y += 1

            cnt_y = 0
            # remove plots not used
            for i in range(ndim_x):
                for j in range(cnt_y,ndim_x):
                    axes[i,j].set_visible(False)
                cnt_y += 1
            #figure = corner.corner(rev_x[:,:5], labels=[r"$Amplitude$", r"$t0$", r"$f$", r"$\phi$", r"$\sigma$"],
            #                       quantiles=[0.68, 0.9, 0.99], color='red',
            #                       show_titles=True, title_kwargs={"fontsize": 12})
            #corner.corner(true_post[i,j,:], labels=[r"$Amplitude$", r"$t0$", r"$f$", r"$\phi$", r"$\sigma$"],
            #                       quantiles=[0.68, 0.9, 0.99], fig=figure, color='blue',
            #                       show_titles=True, title_kwargs={"fontsize": 12}) 
            fig_post.canvas.draw()
            fig_post.tight_layout()
            fig_post.savefig('%sposteriors_%s.pdf' % (out_dir,epoch),dpi=360)
            fig_post.savefig('%slatest.pdf' % out_dir,dpi=360)
            #plt.clf()
            #plt.close()
except KeyboardInterrupt:
    pass
finally:
    print(f"\n\nTraining took {(time()-tStart)/60:.2f} minutes\n")


exit()
# Test the output of the model. The first number is the L2 on the forward process generated line profiles, while the second is the MMD between atmosphere generated by backwards model and the expected atmosphere (and padding).

# In[ ]:

trainer.test(maxBatches=-1)[:2]


# Define functions to allows us to save and load the model and associated machinery in a way that allows us to continue training if desired.

# In[ ]:


# https://discuss.pytorch.org/t/saving-and-loading-a-model-in-pytorch/2610/4
def training_checkpoint():
    return {
        'epoch': totalEpochs,
        'state_dict': model.state_dict(),
        'optimizer': trainer.optim.state_dict(),
        'scheduler': trainer.scheduler.state_dict(),
        'inRepr': inRepr,
        'outRepr': outRepr
    }

def save_checkpoint(state, filename='checkpoint.pth.tar'):
    torch.save(state, filename)

def load_checkpoint(filename):
        if os.path.isfile(filename):
            print("=> loading checkpoint '{}'".format(filename))
            checkpoint = torch.load(filename)
            global totalEpochs
            totalEpochs = checkpoint['epoch']
            model.load_state_dict(checkpoint['state_dict'])
            trainer.optim.load_state_dict(checkpoint['optimizer'])
            trainer.scheduler.load_state_dict(checkpoint['scheduler'])
            inRepr = checkpoint['inRepr']
            outRepr = checkpoint['outRepr']
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(filename, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(filename))


# Do repeated iterations of the training, up to 12000 epochs. Save a model every 600 epochs. This takes quite a while and makes a new plot for each of the 600 epoch batches.

# In[ ]:


prevTest = trainer.test(maxBatches=-1)
while True:
    save_checkpoint(training_checkpoint(), filename='checkpt_'+str(totalEpochs)+'_'+str(int(trainer.wPred))+'.pth.tar')
    trainer.numEpochs = 600
    trainer.fadeIn = False
    trainer.wPred += 1000
    
    # Do the training iter --  this is just a horrible copy and paste from above
    losses = []
    lossVec = [[] for _ in range(4)]
    lossLabels = ['L2 Line', 'MMD Latent', 'MMD Reverse', 'L2 Reverse']
    out = None
    fig, axis = plt.subplots(4,1, figsize=(10,8))
    fig.show()
    fig.canvas.draw()
    alphaRange, mmdF, mmdB, idxF, idxB = [1,1], [1,1], [1,1], 0, 0
    try:
        tStart = time()
        for epoch in range(trainer.numEpochs):
            totalEpochs += 1

            trainer.scheduler.step()

            loss, indLosses = trainer.train(epoch)

            axis[0].clear()
            axis[1].clear()
            axis[2].clear()
            axis[3].clear()
            if epoch > 5:
                for i in range(len(indLosses)):
                    lossVec[i].append(indLosses[i])
                losses.append(loss)
                fig.suptitle('Current Loss: %.2e, min loss: %.2e' % (loss, np.nanmin(np.abs(losses))))
                axis[0].semilogy(np.arange(len(losses)), np.abs(losses))
                for i, lo in enumerate(lossVec):
                    axis[1].semilogy(np.arange(len(losses)), lo, '--', label=lossLabels[i])
                axis[1].legend(loc='upper left')
                tNow = time()
                elapsed = int(tNow - tStart)
                eta = int((tNow - tStart) / (epoch + 1) * trainer.numEpochs) - elapsed

                if epoch % 2 == 0:
                    mses = trainer.test(maxBatches=1)
                    lineProfiles = mses[2]

                if epoch % 10 == 0:
                    alphaRange, mmdF, mmdB, idxF, idxB = trainer.review_mmd()

                axis[3].semilogx(alphaRange, mmdF, label='Latent Space')
                axis[3].semilogx(alphaRange, mmdB, label='Backward')
                axis[3].semilogx(alphaRange[idxF], mmdF[idxF], 'ro')
                axis[3].semilogx(alphaRange[idxB], mmdB[idxB], 'ro')
                axis[3].legend()

                testTime = time() - tNow
                axis[2].plot(lineProfiles[0, model.outSchema.Halpha].cpu().numpy())
                axis[2].plot(lineProfiles[0, model.outSchema.Ca8542].cpu().numpy())
                for a in axis:
                    a.grid()
                axis[3].set_xlabel('Epochs: %d, Elapsed: %d s, ETA: %d s (Testing: %d s)' % (epoch, elapsed, eta, testTime))


            fig.canvas.draw()

    except KeyboardInterrupt:
        pass
    finally:
        print(f"\n\nTraining took {(time()-tStart)/60:.2f} minutes\n")
        
    test = trainer.test(maxBatches=-1)
    print(test[0], test[1])
    
    if totalEpochs >= 12000:
        save_checkpoint(training_checkpoint(), filename='checkpt_'+str(totalEpochs)+'_'+str(int(trainer.wPred))+'.pth.tar')
        break
    


# Loop over all the checkpoint files in the current directory, and compute their accuracy on the unseen testing set

# In[ ]:


files = [f for f in os.listdir() if f.startswith('checkpt_') and f.endswith('.pth.tar')]
numerical = [int(f.split('_')[1]) for f in files]
files = [f[1] for f in sorted(zip(numerical, files))]

for f in files:
    load_checkpoint(f)
    print(trainer.test(maxBatches=-1)[:2])


# This cell can be used to load desired model from the information produced by the previous cell. Just change the argument to numerical.index to the number of epochs the desirec checkpoint was trained for. It will spit out the losses again.

# In[ ]:


files = [f for f in os.listdir() if f.startswith('checkpt_') and f.endswith('.pth.tar')]
numerical = [int(f.split('_')[1]) for f in files]
idx = numerical.index(11400)
load_checkpoint(files[idx])
trainer.test(maxBatches=-1)[:2]


# Define a function to transform from out log-ish velocity to km/s

# In[10]:


def logvel_to_vel(v):
    vSign = v / torch.abs(v)
    vSign[torch.isnan(vSign)] = 0
    vel = vSign * (10**torch.abs(v) - 1.0)
    return vel


# Define a little helper class to better format matplotlib's offset on the x-axis for the huge numbers of cm we're using here!

# In[90]:


import matplotlib.ticker
class oom_formatter(matplotlib.ticker.ScalarFormatter):
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


# Test a random unseen atmosphere with the forward model and compare against the validation data with a nice plot. You may need to run this (quite) a few times to get an atmosphere that produces the line shapes you were looking for. This should produce a different result every time.

# In[94]:


model.eval()
with torch.no_grad():
    x, y = next(iter(data.testLoader))
    x = x.to(dev)
    pad_fn = lambda *x: torch.zeros(*x, device=dev)
    inp = model.inSchema.fill({'ne': x[:, 0],
                                'temperature': x[:, 1],
                                'vel': x[:, 2]},
                               zero_pad_fn=pad_fn)
    
    yz = model(inp.to(dev))
    fig, ax = plt.subplots(2,2, figsize=(8,6))
    ax = ax.ravel()
    ax = [ax[0], ax[0].twinx(), *ax[1:]]
    ax[0].plot(data.z.numpy(), x[0, 0].cpu().numpy())
    ax[1].plot(data.z.numpy(), x[0, 1].cpu().numpy(), color='C1')
    ax[2].plot(data.z.numpy(), logvel_to_vel(x[0, 2].cpu()).numpy(), color='green')
    ax[3].plot(data.wls[0].numpy(), y[0, 0].numpy(), '--', zorder=3)
    ax[3].plot(data.wls[0].numpy(), yz[0, model.outSchema.Halpha].cpu().numpy())
    ax[4].plot(data.wls[1].numpy(), y[0, 1].numpy(), '--', zorder=3, label='Ground Truth')
    ax[4].plot(data.wls[1].numpy(), yz[0, model.outSchema.Ca8542].cpu().numpy(), label='Predicted')
    ax[0].set_ylabel('log $n_e$ [cm$^{-3}$]', color='C0')
    ax[1].set_ylabel('log T [K]', color='C1')
    ax[2].set_ylabel('v [km s$^{-1}$]', color='C2')
    ax[3].set_ylabel('Normalised Intensity')
    ax[0].set_xlabel('z [cm]')
    ax[2].set_xlabel('z [cm]')
    ax[3].set_xlabel(r'Wavelength [$\AA$]')
    ax[4].set_xlabel(r'Wavelength [$\AA$]')
    ax[3].set_title(r'H$_\alpha$')
    ax[4].set_title(r'Ca II 8542$\AA$')
    ax[0].xaxis.set_major_formatter(oom_formatter(8))
    ax[2].xaxis.set_major_formatter(oom_formatter(8))
    ax[3].xaxis.set_major_locator(plt.MaxNLocator(5))
    ax[4].xaxis.set_major_locator(plt.MaxNLocator(5))
    fig.legend(loc='center', frameon=False)
    fig.tight_layout()
    fig.show()
    fig.canvas.draw()


# In[95]:


# Save the above figure if desired
fig.savefig('ForwardProcess2.pdf', dpi=300)


# Test the model's inverse solution on a random validation sample from the test set, with `batchSize` number of random draws from the latent space, plot these results and the round-trip line profiles. The interpretation of these figures is discussed in the paper, but in short, the bars on the 2D histogram for the atmospheric profiles show the probability of the parameter value at each atmospheric node. The dashed black lines show the expected solution. The thin bars on the line profiles show the round trip (i.e. forward(inverse(lineProfiles))) in histogram form.

# In[106]:


get_ipython().run_cell_magic('time', '', "from matplotlib.colors import LogNorm, PowerNorm, LinearSegmentedColormap\n    \nmodel.eval()\nwith torch.no_grad():\n    x, y = next(iter(data.testLoader))\n    batchSize = 40000\n    y = torch.ones((batchSize, *y.shape[1:])) * y[0, :, :]\n    y = y.to(dev)\n    randn = lambda *x: torch.randn(*x, device=dev)\n    yz = model.outSchema.fill({'Halpha': y[:, 0], 'Ca8542': y[:, 1], 'LatentSpace': randn}, zero_pad_fn=pad_fn)\n    xOut = model(yz.to(dev), rev=True)\n    \n    yzRound = model(xOut)\n    zEdges = [data.z[0] - 0.5 * (data.z[1] - data.z[0])]\n    for i in range(data.z.shape[0] - 1):\n        zEdges.append(0.5 * (data.z[i] + data.z[i+1]))\n    zEdges.append(data.z[-1] + 0.5 * (data.z[-1] - data.z[-2]))\n    \n    wlHaEdges = [data.wls[0][0] - 0.5 * (data.wls[0][1] - data.wls[0][0])]\n    for i in range(data.wls[0].shape[0] - 1):\n        wlHaEdges.append(0.5 * (data.wls[0][i] + data.wls[0][i+1]))\n    wlHaEdges.append(data.wls[0][-1] + 0.5 * (data.wls[0][-1] - data.wls[0][-2]))\n    \n    wlCaEdges = [data.wls[1][0] - 0.5 * (data.wls[1][1] - data.wls[1][0])]\n    for i in range(data.wls[1].shape[0] - 1):\n        wlCaEdges.append(0.5 * (data.wls[1][i] + data.wls[1][i+1]))\n    wlCaEdges.append(data.wls[1][-1] + 0.5 * (data.wls[1][-1] - data.wls[1][-2]))\n    \n    neEdges = np.linspace(8, 15, num=101)\n    tEdges = np.linspace(3, 8, num=101)\n    minVel = np.min(np.median(xOut[:, model.inSchema.vel], axis=0))\n    minVel = np.sign(minVel) * 2 * np.abs(minVel) if minVel <= 0 else 0.9 * minVel\n    maxVel = 2 * np.max(np.median(logvel_to_vel(xOut[:, model.inSchema.vel]), axis=0))\n    velEdges = np.linspace(minVel, maxVel, num=101)\n    \n    haIntEdges = np.linspace(0.9 * np.min(np.median(yzRound[:, model.outSchema.Halpha], axis=0)), 1.1 * np.max(np.median(yzRound[:, model.outSchema.Halpha], axis=0)), num=201)\n    caIntEdges = np.linspace(0.9 * np.min(np.median(yzRound[:, model.outSchema.Ca8542], axis=0)), 1.1 * np.max(np.median(yzRound[:, model.outSchema.Ca8542], axis=0)), num=201)\n    \n    cmapNe = [(1.0,1.0,1.0,0.0), (51/255, 187/255, 238/255, 1.0)]\n    neColors = LinearSegmentedColormap.from_list('ne', cmapNe)\n    cmapTemp = [(1.0,1.0,1.0,0.0), (238/255, 119/255, 51/255, 1.0)]\n    tempColors = LinearSegmentedColormap.from_list('temp', cmapTemp)\n    cmapVel = [(1.0,1.0,1.0,0.0), (238/255, 51/255, 119/255, 1.0)]\n    velColors = LinearSegmentedColormap.from_list('vel', cmapVel)\n\n        \n    fig, ax = plt.subplots(2, 2, figsize=(9,7))\n    ax1 = ax[0,0].twinx()\n    ax = ax.ravel()\n    ax = [*ax[:3], ax[2].twinx(), *ax[3:]]\n    \n    ax[0].plot(data.wls[0].numpy(), yz[0, model.outSchema.Halpha].cpu().numpy(), '--', zorder=3)\n    ax[1].plot(data.wls[1].numpy(), yz[0, model.outSchema.Ca8542].cpu().numpy(), '--', zorder=3)\n    \n    powerIdx = 0.3\n    ax[0].hist2d(torch.cat([data.wls[0]] * yzRound.shape[0]).numpy(), yzRound[:, model.outSchema.Halpha].cpu().numpy().reshape((-1,)), bins=(wlHaEdges, haIntEdges), cmap='gray_r', norm=PowerNorm(powerIdx))\n    ax[1].hist2d(torch.cat([data.wls[1]] * yzRound.shape[0]).numpy(), yzRound[:, model.outSchema.Ca8542].cpu().numpy().reshape((-1,)), bins=(wlCaEdges, caIntEdges), cmap='gray_r', norm=PowerNorm(powerIdx))\n    ax[2].hist2d(torch.cat([data.z] * xOut.shape[0]).numpy(), xOut[:, model.inSchema.ne].cpu().numpy().reshape((-1,)), bins=(zEdges, neEdges), cmap=neColors, norm=PowerNorm(powerIdx))\n    ax[3].hist2d(torch.cat([data.z] * xOut.shape[0]).numpy(), xOut[:, model.inSchema.temperature].cpu().numpy().reshape((-1,)), bins=(zEdges, tEdges), cmap=tempColors, norm=PowerNorm(powerIdx))\n    ax[4].hist2d(torch.cat([data.z] * xOut.shape[0]).numpy(), logvel_to_vel(xOut[:, model.inSchema.vel].cpu()).numpy().reshape((-1,)), bins=(zEdges, velEdges), cmap=velColors, norm=PowerNorm(powerIdx))\n    \n    ax[2].plot(data.z.numpy(), x[0, 0].numpy(), 'k--')\n    ax[3].plot(data.z.numpy(), x[0, 1].numpy(), 'k--')\n    ax[4].plot(data.z.numpy(), logvel_to_vel(x[0, 2]).numpy(), 'k--')\n    \n    ax[0].set_title(r'H$\\alpha$')\n    ax[1].set_title(r'Ca II 8542$\\AA$')\n    ax[0].set_xlabel(r'Wavelength [$\\AA$]')\n    ax[1].set_xlabel(r'Wavelength [$\\AA$]')\n    ax[0].set_ylabel(r'Normalised Intensity')\n    \n    ax[2].set_xlabel('z [cm]')\n    ax[4].set_xlabel('z [cm]')\n    ax[2].set_ylabel(r'log $n_e$ [cm$^{-3}$]', color=(cmapNe[-1]))\n    ax[3].set_ylabel(r'log T [K]', color=(cmapTemp[-1]))\n    ax[4].set_ylabel(r'v [km s$^{-1}$]', color=(cmapVel[-1]))\n    ax[2].xaxis.set_major_formatter(oom_formatter(8))\n    ax[4].xaxis.set_major_formatter(oom_formatter(8))\n    fig.tight_layout()")


# In[107]:


# Save the above figure if desired
fig.savefig('InverseProcess2.png', dpi=300)

