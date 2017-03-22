import numpy as np
import os,sys,glob,math
import h5py as h5

from PAL2 import PALmodels, PALutils, pputils as pp


def run_noise_analysis(dataset, niter=1000000):

    h5filename = '../data/simulated_data/' + dataset + '/sim.hdf5'

    # run the common red noise analysis
    print 'Running the common red noise analysis for the simulated dataset...'
    chaindir = '../data/chains/' + dataset
    print 'Noise chains saved in directory ' + chaindir

    os.system('PAL2_run.py --h5File {0} --pulsar all \
              --outDir {1}/fix_spec_nf_30/ --niter {2} --fixWhite \
              --noVaryNoise --mark9 --incRed --incGWB --Tspan 0 --noCorrelations --nf 30 \
              --fixSi 4.33'.format(h5filename, chaindir, niter))

    # run the individual noise analyses for each pulsar
    print 'Running the individual white noise analyses for each pulsar...'
    pfile = h5.File(h5filename)
    psrs = list(pfile.keys())

    for psr in psrs:
        os.system('PAL2_run.py --h5File {0} --pulsar {1} \
                  --outDir chains/{2}/noise/{1}/ --niter {3} \
                  --mark9 --incRed --nf 30'.format(h5filename, psr, dataset, niter))

    # make noise files for each pulsar
    noisedir = '../data/noisefiles/' + dataset
    os.system('mkdir -p {0}'.format(noisedir))
    print 'Making the noise files for each pulsar in directory ' + noisedir
    for psr in psrs:
        cp = pp.ChainPP('chains/{0}/noise/{1}/'.format(dataset, psr))
        ml = cp.get_ml_values(mtype='marg')
        noisefile = noisedir + '/{0}_noise.txt'.format(psr)
        with open(noisefile, 'w') as f:
            for key, val in ml.items():
                f.write('%s %g\n'%(key, val))

    print 'Finished!'


# initialize the optimal statistic using the noise values from file
# from Justin Ellis
def init_os(h5file, psrlist, nf, noisedir=None, noVaryNoise=False,
            incJitterEquad=True, incEquad=True, outputfile=None):

    model = PALmodels.PTAmodels(h5file, pulsars=psrlist)

    # make model dictionary, can play with number of frequencies here
    # Note: Tmax=0 means it will use the largest time span in the array to set the frequency bin size
    fullmodel = model.makeModelDict(incRedNoise=True, incEquad=incEquad, incJitterEquad=incJitterEquad,
                                    likfunc='mark9', incGWB=True, nfreqs=nf, Tmax=0)

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


# compute the optimal statistic, marginalizing over the noise parameters
# noise parameters are drawn from chainfile
def compute_optstat_marg(dataset, psrlist, nf, nreal=1000,
                         noVaryNoise=False, incJitterEquad=True, incEquad=True):
    
    outputdir = '../data/optstat/' + dataset
    if not os.path.exists(outputdir):
        os.makedirs(outputdir)
    
    print 'Initializing model...'
    # initialize the model
    noisedir = '../data/noisefiles/' + dataset + '/'
    h5file = '../data/simulated_data/' + dataset + '/sim.hdf5'

    model = init_os(h5file, psrlist, int(nf), noisedir,
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
    
    for ii in range(nreal):
        preal = chain[np.random.randint(0, chain.shape[0]), :]
        preal = np.concatenate((preal, np.array([4.33])))
        pars[ii,:] = preal[:-1]
        _, _, _, opts[ii], sigs[ii] = model.opt_stat_mark9(preal, fixWhite=True)
    
    # write output to a file (optimal statistic, sigma, and SNR for each noise realization)
    outputfile = outputdir + '/marg_os.dat'
    print 'Writing output to file ' + outputfile
    f = open(outputfile, 'w')
    for i in range(nreal):
        f.write('{0:>13.6e}  {1:>13.6e}  {2:>13.6e}\n'.format(opts[i], sigs[i], opts[i]/sigs[i]))
    f.close()

    return (opts, sigs, opts/sigs)

