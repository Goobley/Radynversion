import numpy as np
import torch
import torch.utils.data
import SG5d

def generate(tot_dataset_size,model='slope',ndata=8,sigma=0.1,prior_bound=[0,1,0,1],seed=0,multi_par=False,dt=0,fnyq=0):

    np.random.seed(seed)
    N = tot_dataset_size

    if model=='slope':

        # draw gradient and intercept from prior
        pars = np.random.uniform(0,1,size=(N,2))
        pars[:,0] = prior_bound[0] + (prior_bound[1]-prior_bound[0])*pars[:,0]
        pars[:,1] = prior_bound[2] + (prior_bound[3]-prior_bound[2])*pars[:,1]

        # make y = mx + c + noise
        noise = np.random.normal(loc=0.0,scale=sigma,size=(N,ndata))
        xvec = np.arange(ndata)/float(ndata)
        sig = np.array([pars[:,0]*x + pars[:,1] for x in xvec]).transpose()
        data = sig + noise
        #data = np.array([pars[:,0]*x + pars[:,1] + n for x,n in zip(xvec,noise)]).transpose()

    elif model=='sg':

        if multi_par == True:
            # set prior bound. TODO: make this not hardcoded

            # draw gradient and intercept from prior
            pars = np.random.uniform(0,1,size=(N,3))
            pars[:,0] = prior_bound[0] + (prior_bound[1]-prior_bound[0])*pars[:,0] # amplitude
            pars[:,1] = prior_bound[2] + (prior_bound[3]-prior_bound[2])*pars[:,1] # t0
            #pars[:,2] = prior_bound[4] + (prior_bound[5]-prior_bound[4])*pars[:,2] # frequency
            #pars[:,3] = prior_bound[6] + (prior_bound[7]-prior_bound[6])*pars[:,3] # phase
            w=3.8
            p=0.5
            pars[:,2] = prior_bound[8] + (prior_bound[9]-prior_bound[8])*pars[:,2] # tau

     
            # make y = Asin()*exp()  + noise
            noise = np.random.normal(loc=0.0,scale=sigma,size=(N,ndata))
            xvec = np.arange(ndata)/float(ndata)
            sig = np.array([pars[:,0]*np.sin(2.0*np.pi*w*fnyq*(x-pars[:,1]) + 2.0*np.pi*p)*np.exp(-((x-pars[:,1])/pars[:,2])**2) for x in xvec]).transpose()
            data = sig + noise
        else:
            # draw gradient and intercept from prior
            pars = np.random.uniform(0,1,size=(N,2))
            pars[:,0] = prior_bound[0] + (prior_bound[1]-prior_bound[0])*pars[:,0] # amp
            pars[:,1] = prior_bound[2] + (prior_bound[3]-prior_bound[2])*pars[:,1] # t0
            w = 6.0*np.pi
            p = 1.0
            tau = 0.25

            # make y = Asin()*exp()  + noise
            noise = np.random.normal(loc=0.0,scale=sigma,size=(N,ndata))
            xvec = np.arange(ndata)/float(ndata)
            sig = np.array([pars[:,0]*np.sin(2.0*np.pi*w*fnyq*(x-pars[:,1]) + p)*np.exp(-((x-pars[:,1])/tau)**2) for x in xvec]).transpose()
            data = sig + noise

    else:
        print('Sorry no model of that name')
        exit(1)

    # randomise the data 
    shuffling = np.random.permutation(N)
    pars = torch.tensor(pars[shuffling], dtype=torch.float)
    data = torch.tensor(data[shuffling], dtype=torch.float)
    sig = torch.tensor(sig[shuffling], dtype=torch.float)

    return pars, data, xvec, sig

def get_lik(ydata,ypars,n_grid=64,sig_model='sg',sigma=None,xvec=None,bound=[0,1,0,1],multi_par=False,N_samp=2500):

    mcx = np.linspace(bound[0],bound[1],n_grid)              # vector of mu values amplitude
    mcy = np.linspace(bound[2],bound[3],n_grid)
    dmcx = mcx[1]-mcx[0]                       # mu spacing
    dmcy = mcy[1]-mcy[0]
    mv, cv = np.meshgrid(mcx,mcy)        # combine into meshed variables

    res = np.zeros((n_grid,n_grid))
    if sig_model=='slope':
        for i,c in enumerate(mcy):
            res[i,:] = np.array([np.sum(((ydata-m*xvec-c)/sigma)**2) for m in mcx])
        res = np.exp(-0.5*res)
    elif sig_model=='sg' and multi_par==False:
        w = 6.0*np.pi
        p = 1.0
        tau = 0.25
        # xvec is length of data
        for i,t in enumerate(mcy):
            res[i,:] = np.array([np.sum(((ydata - A*np.sin(w*xvec + p)*np.exp(-((xvec-t)/tau)**2))/sigma)**2) for A in mcx])
        res = np.exp(-0.5*res)

        # get points sampled from the posterior
        post_points = SG5d.run(ydata,ypars,sigma,xvec,np.log(res),multi_par,bound)

    elif sig_model=='sg' and multi_par==True:
        #w = np.linspace(bound[4],bound[5],n_grid)
        #p = np.linspace(bound[6],bound[7],n_grid)
        w=3.8
        p=0.5
        tau = np.linspace(bound[8],bound[9],n_grid)

        # spacing
        #dw = w[1]-w[0]
        #dp = p[1]-p[0]
        dtau = tau[1]-tau[0]

        # xvec is length of data
        for i,t in enumerate(mcy):
            res[i,:] = np.array([np.sum(((ydata - A*np.sin(w*xvec + p)*np.exp(-((xvec-t)/tau[i])**2))/sigma)**2) for A in mcx])
        res = np.exp(-0.5*res)

        # get points sampled from the posterior
        post_points = SG5d.run(ydata,ypars,sigma,xvec,np.log(res),multi_par,bound)

    # normalise the posterior
    res /= (np.sum(res.flatten())*dmcx*dmcy)

    # compute integrated probability outwards from max point
    res = res.flatten()
    idx = np.argsort(res)[::-1]
    prob = np.zeros(n_grid*n_grid)
    prob[idx] = np.cumsum(res[idx])*dmcx*dmcy
    prob = prob.reshape(n_grid,n_grid)
    res = res.reshape(n_grid,n_grid)
    return mcx, mcy, prob, post_points


