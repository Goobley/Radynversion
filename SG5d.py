import numpy as np
import emcee
import corner
import matplotlib.pyplot as plt

# general parameters
#N = 128          # samples in timeseries
#T = 1.0         # length of timeseries (sec)
#sig = 0.1       # standard deviation of the noise
#dt = T/N        # sampling time (Sec)
#fnyq = 0.5/dt   # Nyquist frequency (Hz)

# make tim vector
#t = np.arange(N)*dt

# define true parameters
#theta_true = np.array([0.8,0.4,0.1,0.2,0.2])
labels = ["$A$", "$t_0$", "f_0", "$\\phi$", "\\tau"]

def sg(x,theta):
    """
    waveform generator - Sine-Gaussian 
    A = amplitude 
    t0 = central time (sec)
    phi = phase at t0 (in cycles NOT rads)
    f0 = frequency (as fraction of Nyquist frequency)
    tau = exponential envelope decay time (sec)
    ALL range from 0 - 1
    """
    #A,t0,f0,phi,tau = theta
    A,t0,tau = theta
    f0=3.8
    phi=0.5
    return A*np.sin(2.0*np.pi*f0*fnyq*(x-t0) + 2.0*np.pi*phi)*np.exp(-((x-t0)/tau)**2)

def lnprior(theta):
    """
    returns 0 probability if outside the prior bounds
    and 1 if inside for ALL parameters
    """
    #A,t0,f0,phi,tau = theta
    A,t0,tau = theta
    if np.any(theta>1.0) or np.any(theta<0.0):
        return -np.inf
    return 0.0

def lnlike(theta, x, y, yerr):
    """
    returns the log-likelihood ignoring the constant factors
    from Gaussian normalisation
    x = time vector (sec)
    y = data
    yerr = standard deviation of the noise
    """
    return -0.5*np.sum(((y-sg(x,theta))/yerr)**2)

def lnprob(theta, x, y, yerr):
    """
    combines log-likelihood and log-prior
    """
    lp = lnprior(theta)
    if not np.isfinite(lp):
        return -np.inf
    return lp + lnlike(theta, x, y, yerr)

def run(ydata,ypars,sigma,x,loglikelihood,multi_par,bounds):
    # make the data and plot it
    #s = sg(t,theta_true) # equivalent to ydata
    #n = np.random.normal(0,sig,N) # noise
    #h = s + n # signal plus noise
    #plt.figure()
    #plt.plot(t,h)
    #plt.plot(t,s)
    #plt.xlabel('time (sec)')
    #plt.savefig('/home/hunter.gabbard/public_html/CBC/cINNamon/gausian_results/multipar/test_data.png')
    #plt.close()

    h=ydata
    theta_true=ypars

    T = 1.0
    dt = T/len(x)        # sampling time (Sec)
    global fnyq
    fnyq = 0.5/dt   # Nyquist frequency (Hz)
    # number of walkers and length of each walk must be large for 5D
    ndim, nwalkers, M = ypars.shape[0], 50, 10000

    # run emcee initial positions unformly distributed inside the unit cube
    #pos = [np.random.uniform(0,1,ndim) for i in range(nwalkers)]
    amin = bounds[0]     # lower range of prior
    amax = bounds[1] # upper range of prior

    aini = np.random.uniform(amin, amax, nwalkers) # initial amplitude

    t_0min = bounds[2]  # lower range of prior
    t_0max = bounds[3]   # upper range of prior

    t_0ini = np.random.uniform(t_0min, t_0max, nwalkers) # initial t0 points

    if multi_par==True:
        #fmin = bounds[4]
        #fmax = bounds[5]

        #fini = np.random.uniform(fmin, fmax, nwalkers) # initial frequency points

        #pmin = bounds[6]
        #pmax = bounds[7]

        #pini = np.random.uniform(pmin, pmax, nwalkers) # initial phase points

        tmin = bounds[8]*0.01
        tmax = bounds[9]*0.01

        tini = np.random.uniform(tmin, tmax, nwalkers) # initial f points

        pos = np.array([aini, t_0ini, tini]).T # initial samples

    else: pos = np.array([aini, t_0ini]).T # initial samples

    pos = [np.random.uniform(0,1,ndim) for i in range(nwalkers)]
    #sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob, args=(t, h, sig))
    sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob, args=(x, h, sigma))
    sampler.run_mcmc(pos, M)

    # remove all bad points (with log-likelihood e^12 times less than the max)
    all_samples = sampler.chain[:, :, :].reshape((-1, ndim))
    all_lnp = sampler.lnprobability[:,:].flatten()
    max_lnp = np.max(all_lnp)
    idx = np.argwhere(all_lnp>max_lnp-12.0).squeeze()
    samples = all_samples[idx,:]

    # plot the remaining chains - no need for burn-in
    for i in range(ndim):
        plt.figure()
        plt.plot(samples[:,i],'.',markersize=0.2)
        plt.plot([0,len(idx)],[theta_true[i],theta_true[i]],'-k')
        plt.ylim([0,1])
        #plt.hist(samples[:,i], bins=100)
        plt.ylabel(labels[i])
        plt.xlabel('sample')
        plt.savefig('/home/hunter.gabbard/public_html/CBC/cINNamon/gausian_results/multipar/test_chain_%s.png' % str(i))
        plt.close()

   
    #make corner plot - use 5000 randomly slected samples
    #idx = np.random.randint(0,samples.shape[0],5000)
    #fig = corner.corner(samples[idx], labels=labels, truths=theta_true)
    #plt.savefig('/home/hunter.gabbard/public_html/CBC/cINNamon/gausian_results/multipar/test_corner.png')
    #plt.close()

    print('Number of posterior samples is {}'.format(samples.shape[0]))
    print("Mean acceptance fraction: {0:.3f}"
                .format(np.mean(sampler.acceptance_fraction)))
    return samples
