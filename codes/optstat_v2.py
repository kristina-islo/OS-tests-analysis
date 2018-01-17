import numpy as np
import os,sys,glob,math
import h5py as h5
import scipy.interpolate as interp
import scipy.ndimage.filters as filter

from PAL2 import PALmodels, PALutils, pputils as pp

import makesims_nano


def get_ml_vals(cp):

    ind = np.arange(0, cp.ndim)
    ind = np.atleast_1d(ind)

    x = OrderedDict()

    for i in ind:
        x[cp.pars[i]] = bu.getMax(cp.chain[:,i])

    return x


def getMax2d(samples1, samples2, weights=None, smooth=True, bins=[40, 40],
            x_range=None, y_range=None, logx=False, logy=False, logz=False):
    
    if x_range is None:
        xmin = np.min(samples1)
        xmax = np.max(samples1)
    else:
        xmin = x_range[0]
        xmax = x_range[1]

    if y_range is None:
        ymin = np.min(samples2)
        ymax = np.max(samples2)
    else:
        ymin = y_range[0]
        ymax = y_range[1]

    if logx:
        bins[0] = np.logspace(np.log10(xmin), np.log10(xmax), bins[0])
    
    if logy:
        bins[1] = np.logspace(np.log10(ymin), np.log10(ymax), bins[1])

    hist2d,xedges,yedges = np.histogram2d(samples1, samples2, weights=weights, \
            bins=bins,range=[[xmin,xmax],[ymin,ymax]])
    extent = [xedges[0], xedges[-1], yedges[0], yedges[-1] ]

    if logz:
        for ii in range(hist2d.shape[0]):
            for jj in range(hist2d.shape[1]):
                if hist2d[ii,jj] <= 0:
                    hist2d[ii,jj] = 1

    
    xedges = np.delete(xedges, -1) + 0.5*(xedges[1] - xedges[0])
    yedges = np.delete(yedges, -1) + 0.5*(yedges[1] - yedges[0])
    
    # gaussian smoothing
    if smooth:
        hist2d = filter.gaussian_filter(hist2d, sigma=0.75)

    # interpolation
    f = interp.interp2d(xedges, yedges, hist2d, kind='cubic')
    xedges = np.linspace(xedges.min(), xedges.max(), 10000)
    yedges = np.linspace(yedges.min(), yedges.max(), 10000)
    hist2d = f(xedges, yedges)

    # return xedges[np.argmax(hist2d)]
    ind = np.unravel_index(np.argmax(hist2d), hist2d.shape)
    return xedges[ind[0]], yedges[ind[1]]


def run_noise_analysis(dataset, writenoise, niter=1000000):

    h5filename = '../data/simulated_data/' + dataset + '/sim.hdf5'

    # # run the common red noise analysis
    print 'Running the common red noise analysis for the simulated dataset...'
    chaindir = '../data/chains/' + dataset
    print 'Noise chains saved in directory ' + chaindir

    os.system('PAL2_run.py --h5File {0} --pulsar all \
              --outDir {1}/fix_spec_nf_30/ --niter {2} --fixWhite \
              --noVaryNoise --mark9 --incRed --incGWB --Tspan 0 --noCorrelations --nf 30 \
              --fixSi 4.33'.format(h5filename, chaindir, niter))

    # run the individual noise analyses for each pulsar
    print 'Running the individual white noise analyses for each pulsar...'

    with h5.File(h5filename) as f:
        psrs = list(f.keys())
    print psrs

    for psr in psrs:
        print psr
        os.system('PAL2_run.py --h5File {0} --pulsar {1} \
                  --outDir ../data/chains/{2}/noise/{1}/ --niter {3} \
                  --mark9 --incRed --nf 30'.format(h5filename, psr, dataset, niter))

    # make noise files for each pulsar
    noisedir = '../data/noisefiles/' + dataset
    os.system('mkdir -p {0}'.format(noisedir))
    print 'Making the noise files for each pulsar in directory ' + noisedir
    
    for n,psr in enumerate(psrs):
        
        if writenoise == '1dmax':
            # create chain object and find 1d maxLL values
            cp = pp.ChainPP('../data/chains/{0}/noise/{1}/'.format(dataset, psr))
            ml = cp.get_ml_values(mtype='marg')
            noisefile = noisedir + '/{0}_noise.txt'.format(psr)
            with open(noisefile, 'w') as f:
                for key, val in ml.items():
                    f.write('%s %g\n'%(key, val))


        if writenoise == 'maxsample':
            # load chain and parameters names
            pars = list(np.loadtxt('../data/chains/{0}/noise/{1}/pars.txt'.format(dataset, psr), dtype='S42'))
            chain = np.loadtxt('../data/chains/{0}/fix_spec_nf_30/chain_1.txt'.format(dataset))
            burn = int(0.25*chain.shape[0])
            # find max sample
            index = np.argmax(chain[burn:,-3])
            maxpost_sample = chain[index,:]
            # write
            noisefile = noisedir + '/{0}_noise.txt'.format(psr)
            with open(noisefile, 'w') as f:
                for p,val in zip(pars,maxpost_sample):
                    f.write('%s %g\n'%(p, val))

        if writenoise == '2dmax':

            # create chain object and find 1d maxLL parameter values
            cp = pp.ChainPP('../data/chains/{0}/noise/{1}/'.format(dataset, psr))
            try:
                ml = cp.get_ml_values(mtype='marg')
            except:
                # this sometimes arises for J1909
                # 1d maximization for efac
                ind = np.where(cp.pars == 'efac')
                ml = cp.get_ml_values(mtype='marg', ind=ind[0]) 

                # initialise the two RN parameters in ml object
                ml.update({'RN-Amplitude': None})
                ml.update({'RN-spectral-index': None})

            noisefile = noisedir + '/{0}_noise.txt'.format(psr) 

            # load individual chains and find 2d maxLL values for RN parameters
            pars = np.loadtxt('../data/chains/{0}/noise/{1}/pars.txt'.format(dataset, psr), dtype='S42')
            chain = np.loadtxt('../data/chains/{0}/noise/{1}/chain_1.txt'.format(dataset, psr))
            burn = int(0.25*chain.shape[0])


            RN_amplitude_chain = chain[:,np.argwhere(pars == 'RN-Amplitude')[0][0]]
            RN_spectral_index_chain = chain[:,np.argwhere(pars == 'RN-spectral-index')[0][0]]
            RN_Amplitude_individual, RN_spectral_index_individual = getMax2d(RN_amplitude_chain, RN_spectral_index_chain)


            # load common red chain and find 2d maxLL values for RN parameters
            pars = np.loadtxt('../data/chains/{0}/fix_spec_nf_30/pars.txt'.format(dataset), dtype='S42')     
            chain = np.loadtxt('../data/chains/{0}/fix_spec_nf_30/chain_1.txt'.format(dataset))
            burn = int(0.25*chain.shape[0])
    
            RN_amplitude_chain = chain[:,np.argwhere(pars == 'RN-Amplitude_{0}'.format(psr))[0][0]]
            RN_spectral_index_chain = chain[:,np.argwhere(pars == 'RN-spectral-index_{0}'.format(psr))[0][0]]
            RN_Amplitude_common, RN_spectral_index_common = getMax2d(RN_amplitude_chain, RN_spectral_index_chain)


            with open(noisefile, 'w') as f:

                for key, val in ml.items():

                    if key == 'RN-Amplitude':
                        f.write('%s %g %g\n'%(key, RN_Amplitude_individual, RN_Amplitude_common))

                    elif key == 'RN-spectral-index':
                        f.write('%s %g %g\n'%(key, RN_spectral_index_individual, RN_spectral_index_common))

                    else:
                        f.write('%s %g %g\n'%(key, val, val))

    print 'Finished!'


def compute_orf(ptheta, pphi):
    
    npsr = len(ptheta)
    pos = [ np.array([np.cos(phi)*np.sin(theta),
                      np.sin(phi)*np.sin(theta),
                      np.cos(theta)]) for phi, theta in zip(pphi, ptheta) ]
        
    x = []
    for i in range(npsr):
        for j in range(i+1,npsr):
            x.append(np.dot(pos[i], pos[j]))
    x = np.array(x)

    orf = HD(x)
    
    return orf


def compute_monopole(psr):
    
    npsr = len(psr)
    npairs = int(npsr*(npsr-1)/2)
    ORF = np.ones(npairs)
    
    return ORF


def compute_dipole(psr):
    
    npsr = len(psr)
    npairs = int(npsr*(npsr-1)/2)
    ORF = np.zeros(npairs)
    
    phati = np.zeros(3)
    phatj = np.zeros(3)
    
    # begin loop over all pulsar pairs and calculate ORF
    k = 0
    for i in range(npsr):
        phati[0] = np.cos(psr[i].phi) * np.sin(psr[i].theta)
        phati[1] = np.sin(psr[i].phi) * np.sin(psr[i].theta)
        phati[2] = np.cos(psr[i].theta)
        
        for j in range(i+1,npsr):
            phatj[0] = np.cos(psr[j].phi) * np.sin(psr[j].theta)
            phatj[1] = np.sin(psr[j].phi) * np.sin(psr[j].theta)
            phatj[2] = np.cos(psr[j].theta)
            
            costhetaij = np.sum(phati*phatj)
            
            ORF[k] = costhetaij
            k += 1
    
    return ORF


# returns the Hellings and Downs coefficient for two DIFFERENT pulsars
#    x :   cosine of the angle between the pulsars
def HD(x):
    return 1.5*(1./3. + (1.-x)/2.*(np.log((1.-x)/2.)-1./6.))


def compute_os(orf, rho, sig):
    
    opt = np.sum(np.array(rho) * orf / np.array(sig) ** 2) / np.sum(orf ** 2 / np.array(sig) ** 2)
    sig = 1 / np.sqrt(np.sum(orf ** 2 / np.array(sig) ** 2))
    
    return opt, sig


# reads in sky scramble positions from files provided by Steve Taylor
def read_in_skyscrambles(directory, nscrambles):
    
    orfs = []
    
    for n in range(nscrambles):
        data = np.loadtxt(directory + 'PositionSet_{0:d}.dat'.format(n),
                          skiprows=1, usecols=(1,2))
                          
        orfs.append(compute_orf(data[:,1], data[:,0]))
    
    return orfs


# initialize the optimal statistic using the noise values from file
# from Justin Ellis
def init_os_individual(dataset, h5file, psrlist, nf, noisedir=None, noVaryNoise=False,
            incJitterEquad=True, incEquad=True, outputfile=None):

    model = PALmodels.PTAmodels(h5file, pulsars=psrlist)

    # make model dictionary, can play with number of frequencies here
    # Note: Tmax=0 means it will use the largest time span in the array to set the frequency bin size
    fullmodel = model.makeModelDict(incRedNoise=True, incEquad=incEquad, incJitterEquad=incJitterEquad,
                                    likfunc='mark9', incGWB=True, incEphemMixture=False, nfreqs=nf, Tmax=0)

    # loop through pulsars and fix white noise parameters to values in file
    if noisedir is not None:
        for ct, p in enumerate(model.psr):
            d = np.genfromtxt(noisedir + p.name + '_noise.txt', dtype='S42')
            pars = d[:,0]
            vals = np.array([float(d[ii,1]) for ii in range(d.shape[0])])
            sigs = [psig for psig in fullmodel['signals'] if psig['pulsarind'] == ct]
            sigs = PALutils.fixNoiseValues(sigs, vals, pars, bvary=False, verbose=False)
            
            # turn back on red-noise parameters (will use later when drawing from posteriors)
            for sig in fullmodel['signals']:
                if sig['corr'] == 'single' and sig['stype'] == 'powerlaw':
                    sig['bvary'][1] = True
                    sig['bvary'][0] = True

    if noVaryNoise:
        nflags = ['efac', 'equad', 'jitter', 'jitter_equad']
        for sig in fullmodel['signals']:
            if sig['stype'] in nflags:
                sig['bvary'][0] = False

    # intialize mdoel
    model.initModel(fullmodel, memsave=True, write='no')
    
    # start parameters off at initial values (i.e the fixed values red in above)
    p0 = model.initParameters(fixpstart=True)
    
    # essentially turn off GWB component (again we will do something different when drawing from full PTA posterior)
    p0[-2] = -19
    
    # call once to fix white noise
    xi, rho, sig, Opt, Sig = model.opt_stat_mark9(p0)

    if outputfile is not None:
        f = open(outputfile, 'w')
        f.write('Opt  {0}\n'.format(Opt))
        f.write('Sig  {0}\n'.format(Sig))
        f.write('SNR  {0}\n'.format(Opt/Sig))
        f.close()

    return model


def init_os_common(dataset, h5file, psrlist, nf, noisedir=None, noVaryNoise=False,
            incJitterEquad=True, incEquad=True, outputfile=None):

    model = PALmodels.PTAmodels(h5file, pulsars=psrlist)

    # make model dictionary, can play with number of frequencies here
    # Note: Tmax=0 means it will use the largest time span in the array to set the frequency bin size
    fullmodel = model.makeModelDict(incRedNoise=True, incEquad=incEquad, incJitterEquad=incJitterEquad,
                                    likfunc='mark9', incGWB=True, incEphemMixture=False, nfreqs=nf, Tmax=0)

    # loop through pulsars and fix white noise parameters to values in file
    if noisedir is not None:
        for ct, p in enumerate(model.psr):
            d = np.genfromtxt(noisedir + p.name + '_noise.txt', dtype='S42')
            pars = d[:,0]
            vals = np.array([float(d[ii,2]) for ii in range(d.shape[0])])
            sigs = [psig for psig in fullmodel['signals'] if psig['pulsarind'] == ct]
            sigs = PALutils.fixNoiseValues(sigs, vals, pars, bvary=False, verbose=False)
            
            # turn back on red-noise parameters (will use later when drawing from posteriors)
            for sig in fullmodel['signals']:
                if sig['corr'] == 'single' and sig['stype'] == 'powerlaw':
                    sig['bvary'][1] = True
                    sig['bvary'][0] = True

    if noVaryNoise:
        nflags = ['efac', 'equad', 'jitter', 'jitter_equad']
        for sig in fullmodel['signals']:
            if sig['stype'] in nflags:
                sig['bvary'][0] = False

    # intialize mdoel
    model.initModel(fullmodel, memsave=True, write='no')
    
    # start parameters off at initial values (i.e the fixed values red in above)
    p0 = model.initParameters(fixpstart=True)
    
    # essentially turn off GWB component (again we will do something different when drawing from full PTA posterior)
    p0[-2] = -19
    
    # call once to fix white noise
    xi, rho, sig, Opt, Sig = model.opt_stat_mark9(p0)

    if outputfile is not None:
        f = open(outputfile, 'w')
        f.write('Opt  {0}\n'.format(Opt))
        f.write('Sig  {0}\n'.format(Sig))
        f.write('SNR  {0}\n'.format(Opt/Sig))
        f.close()

    return model


# compute the optimal statistic, marginalizing over the noise parameters
# noise parameters are drawn from chainfile
def compute_optstat_marg_individual(dataset, psrlist, nf, nreal=1000,
                         noVaryNoise=False, incJitterEquad=True, incEquad=True,
                         computeMonopole=False, computeDipole=False,
                         computeSkyScrambles=False, skyScrambleDir='.', nscrambles=100):
    
    outputdir = '../data/optstat_individual/' + dataset
    if not os.path.exists(outputdir):
        os.makedirs(outputdir)
    
    print 'Initializing model...'
    # initialize the model
    noisedir = '../data/noisefiles/' + dataset + '/'
    h5file = '../data/simulated_data/' + dataset + '/sim.hdf5'

    model = init_os_individual(dataset, h5file, psrlist, int(nf), noisedir,
                    noVaryNoise=noVaryNoise, incJitterEquad=incJitterEquad, incEquad=incEquad,
                    outputfile=outputdir+'/init_os.dat')

    # load chains with noise parameters
    chainfile = '../data/chains/' + dataset + '/fix_spec_nf_30/chain_1.txt'
    print 'Loading chains from ' + chainfile
    chain = np.loadtxt(chainfile)
    burn = int(0.25*chain.shape[0])
    chain = chain[burn:,:-4]
    
    print 'Running {0} realizations of the noise...'.format(nreal)
    
    opts, sigs = np.zeros(nreal), np.zeros(nreal)
    pars = np.zeros((nreal, chain.shape[1]))

    if computeMonopole:
        xi1 = compute_monopole(model.psr)
        opts1, sigs1 = np.zeros(nreal), np.zeros(nreal)
    if computeDipole:
        xi2 = compute_dipole(model.psr)
        opts2, sigs2 = np.zeros(nreal), np.zeros(nreal)

    if computeSkyScrambles:
        xi_ss = read_in_skyscrambles(skyScrambleDir, nscrambles)
        opt_ss = [ np.zeros(nreal) for i in range(nscrambles) ]
        sig_ss = [ np.zeros(nreal) for i in range(nscrambles) ]

    for ii in range(nreal):
        preal = chain[np.random.randint(0, chain.shape[0]), :]
        preal = np.concatenate((preal, np.array([4.33])))
        pars[ii,:] = preal[:-1]
        xi, rho, sig, opts[ii], sigs[ii] = model.opt_stat_mark9(preal, fixWhite=True)
        if computeMonopole:
            opts1[ii], sigs1[ii] = compute_os(xi1, rho, sig)
        if computeDipole:
            opts2[ii], sigs2[ii] = compute_os(xi2, rho, sig)
        if computeSkyScrambles:
            for j in range(nscrambles):
                opt_ss[j][ii], sig_ss[j][ii] = compute_os(xi_ss[j], rho, sig)
        if ii > 1:
            sys.stdout.write('\r')
            sys.stdout.write('Finished %2.2f percent'
                                 % (ii / nreal * 100)
            sys.stdout.flush()

    # write output to a file (optimal statistic, sigma, and SNR for each noise realization)
    outputfile = outputdir + '/marg_os.dat'
    print 'Writing output to file ' + outputfile
    f = open(outputfile, 'w')
    for i in range(nreal):
        f.write('{0:>13.6e}  {1:>13.6e}  {2:>13.6e}\n'.format(opts[i], sigs[i], opts[i]/sigs[i]))
    f.close()

    if computeMonopole:
        f1 = open(outputdir + '/marg_os_monopole.dat', 'w')
        for i in range(nreal):
            f1.write('{0:>13.6e}  {1:>13.6e}  {2:>13.6e}\n'.format(opts1[i], sigs1[i], opts1[i]/sigs1[i]))
        f1.close()

    if computeDipole:
        f2 = open(outputdir + '/marg_os_dipole.dat', 'w')
        for i in range(nreal):
            f2.write('{0:>13.6e}  {1:>13.6e}  {2:>13.6e}\n'.format(opts2[i], sigs2[i], opts2[i]/sigs2[i]))
        f2.close()

    if computeSkyScrambles:
        for n in range(nscrambles):
            f = open(outputdir + '/marg_os_skyscramble{0}.dat'.format(n), 'w')
            for i in range(nreal):
                f.write('{0:>13.6e}  {1:>13.6e}  {2:>13.6e}\n'.format(opt_ss[n][i],
                                                                      sig_ss[n][i],
                                                                      opt_ss[n][i]/sig_ss[n][i]))
            f.close()

    return (opts, sigs, opts/sigs)


# compute the optimal statistic, marginalizing over the noise parameters
# noise parameters are drawn from chainfile
def compute_optstat_marg_common(dataset, psrlist, nf, nreal=1000,
                         noVaryNoise=False, incJitterEquad=True, incEquad=True,
                         computeMonopole=False, computeDipole=False,
                         computeSkyScrambles=False, skyScrambleDir='.', nscrambles=100):
    
    outputdir = '../data/optstat_common/' + dataset
    if not os.path.exists(outputdir):
        os.makedirs(outputdir)
    
    print 'Initializing model...'
    # initialize the model
    noisedir = '../data/noisefiles/' + dataset + '/'
    h5file = '../data/simulated_data/' + dataset + '/sim.hdf5'

    model = init_os_common(dataset, h5file, psrlist, int(nf), noisedir,
                    noVaryNoise=noVaryNoise, incJitterEquad=incJitterEquad, incEquad=incEquad,
                    outputfile=outputdir+'/init_os.dat')

    # load chains with noise parameters
    chainfile = '../data/chains/' + dataset + '/fix_spec_nf_30/chain_1.txt'
    print 'Loading chains from ' + chainfile
    chain = np.loadtxt(chainfile)
    burn = int(0.25*chain.shape[0])
    chain = chain[burn:,:-4]
    
    print 'Running {0} realizations of the noise...'.format(nreal)
    
    opts, sigs = np.zeros(nreal), np.zeros(nreal)
    pars = np.zeros((nreal, chain.shape[1]))

    if computeMonopole:
        xi1 = compute_monopole(model.psr)
        opts1, sigs1 = np.zeros(nreal), np.zeros(nreal)
    if computeDipole:
        xi2 = compute_dipole(model.psr)
        opts2, sigs2 = np.zeros(nreal), np.zeros(nreal)

    if computeSkyScrambles:
        xi_ss = read_in_skyscrambles(skyScrambleDir, nscrambles)
        opt_ss = [ np.zeros(nreal) for i in range(nscrambles) ]
        sig_ss = [ np.zeros(nreal) for i in range(nscrambles) ]

    for ii in range(nreal):
        preal = chain[np.random.randint(0, chain.shape[0]), :]
        preal = np.concatenate((preal, np.array([4.33])))
        pars[ii,:] = preal[:-1]
        xi, rho, sig, opts[ii], sigs[ii] = model.opt_stat_mark9(preal, fixWhite=True)
        if computeMonopole:
            opts1[ii], sigs1[ii] = compute_os(xi1, rho, sig)
        if computeDipole:
            opts2[ii], sigs2[ii] = compute_os(xi2, rho, sig)
        if computeSkyScrambles:
            for j in range(nscrambles):
                opt_ss[j][ii], sig_ss[j][ii] = compute_os(xi_ss[j], rho, sig)
        if ii > 1:
            sys.stdout.write('\r')
            sys.stdout.write('Finished %2.2f percent'
                                % (ii / nreal * 100))
            sys.stdout.flush()

    # write output to a file (optimal statistic, sigma, and SNR for each noise realization)
    outputfile = outputdir + '/marg_os.dat'
    print 'Writing output to file ' + outputfile
    f = open(outputfile, 'w')
    for i in range(nreal):
        f.write('{0:>13.6e}  {1:>13.6e}  {2:>13.6e}\n'.format(opts[i], sigs[i], opts[i]/sigs[i]))
    f.close()

    if computeMonopole:
        f1 = open(outputdir + '/marg_os_monopole.dat', 'w')
        for i in range(nreal):
            f1.write('{0:>13.6e}  {1:>13.6e}  {2:>13.6e}\n'.format(opts1[i], sigs1[i], opts1[i]/sigs1[i]))
        f1.close()

    if computeDipole:
        f2 = open(outputdir + '/marg_os_dipole.dat', 'w')
        for i in range(nreal):
            f2.write('{0:>13.6e}  {1:>13.6e}  {2:>13.6e}\n'.format(opts2[i], sigs2[i], opts2[i]/sigs2[i]))
        f2.close()

    if computeSkyScrambles:
        for n in range(nscrambles):
            f = open(outputdir + '/marg_os_skyscramble{0}.dat'.format(n), 'w')
            for i in range(nreal):
                f.write('{0:>13.6e}  {1:>13.6e}  {2:>13.6e}\n'.format(opt_ss[n][i],
                                                                      sig_ss[n][i],
                                                                      opt_ss[n][i]/sig_ss[n][i]))
            f.close()

    return (opts, sigs, opts/sigs)


if __name__ == '__main__':
    
    import argparse
    
    parser = argparse.ArgumentParser(description='generates simulated data and computes the optimal statistic for each simulation')

    parser.add_argument('--Agw', default=5e-15,
                        help='GW amplitude (DEFAULT: 5e-15)')
    parser.add_argument('--datasetname', default='dataset',
                        help='name for this data set')
    parser.add_argument('--skipdatacreate', action='store_true')
    parser.add_argument('--skipnoiseanalysis', action='store_true')
    parser.add_argument('--writenoise', default='2dmax',
                        help='how to determine individual pulsar noise values \
                        (1-d maximization / maximum likelihood sample from common red process / 2-d maximization from individual pulsar noise chains & \
                        2-d maximization from common red process chain, DEFAULT: 2dmax)')
    parser.add_argument('--nreal', default=10000, 
                        help='number of realizations (DEFAULT: 10000)')
    parser.add_argument('--computeMonopole', action='store_true',
                        help='also compute optimal statistic for monopole spatial correlations (DEFAULT: false)')
    parser.add_argument('--computeDipole', action='store_true',
                        help='also compute optimal statistic for dipole spatial correlations (DEFAULT: false)')
    parser.add_argument('--computeSkyScrambles', action='store_true',
                        help='also compute optimal statistic for sky scrambles (DEFAULT: false)')
    parser.add_argument('--skyScrambleDir', default='../data/PosSetSim2/',
                        help='directory containing sky scramble positions')
    parser.add_argument('--nscrambles', default=210,
                        help='number of sky scrambles to run (DEFAULT: 210)')
    
    args = parser.parse_args()
    
    Agwb = float(args.Agw)
    dataset = args.datasetname
    writenoise = args.writenoise
    
    if not args.skipdatacreate:
        makesims_nano.create_dataset(dataset, Agwb)
    
    if not args.skipnoiseanalysis:
        run_noise_analysis(dataset, writenoise)
    
    psrlist = list(np.loadtxt('../data/psrList_nano.txt', dtype='S42'))
    h5file = '../data/simulated_data/' + dataset + '/sim.hdf5'
    opts, sigs, snr = compute_optstat_marg_individual(dataset, psrlist, nf=30, 
                                           noVaryNoise=True, incJitterEquad=False, 
                                           incEquad=False,
                                           nreal=int(args.nreal), 
                                           computeMonopole=args.computeMonopole,
                                           computeDipole=args.computeDipole,
                                           computeSkyScrambles=args.computeSkyScrambles,
                                           skyScrambleDir=args.skyScrambleDir,
                                           nscrambles=int(args.nscrambles))

    opts, sigs, snr = compute_optstat_marg_common(dataset, psrlist, nf=30, 
                                           noVaryNoise=True, incJitterEquad=False, 
                                           incEquad=False,
                                           nreal=int(args.nreal), 
                                           computeMonopole=args.computeMonopole,
                                           computeDipole=args.computeDipole,
                                           computeSkyScrambles=args.computeSkyScrambles,
                                           skyScrambleDir=args.skyScrambleDir,
                                           nscrambles=int(args.nscrambles))
