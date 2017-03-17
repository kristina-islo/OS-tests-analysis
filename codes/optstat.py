import numpy as np
import libstempo as T
import libstempo.toasim as LT
import os,sys,glob,math
from PAL2 import pputils as pp
import h5py as h5
import NX01_psr


# Steve's code to create simulated pulsar timing data with specified properties
def create_dataset(dataset, Agwb, psr, parfiles, pwhite, predAmp, predGam):

    datadir = '../data/simulated_data/' + dataset
    if not os.path.exists(datadir):
        os.makedirs(datadir)

    psrtmp = []
    for ii in range(len(psr)):
    
        psrtmp.append(LT.fakepulsar(parfile = parfiles[ii],
                                    obstimes = psr[ii].toas,
                                    toaerr = psr[ii].toaerrs/1e-6))

    ## Add in pulsar white noise

    for ii in range(len(psrtmp)):
        LT.add_efac(psrtmp[ii], efac=1.0)

    for ii in range(len(psrtmp)):
        psrtmp[ii].fit(iters=5)

    ## Add in red noise with same spectral properties as Sim2

    for ii in range(len(psrtmp)):
        LT.add_rednoise(psrtmp[ii],10.0**predAmp[ii],predGam[ii],
                        components=50,seed=None)

    for ii in range(len(psrtmp)):
        psrtmp[ii].fit(iters=5)

    ## Add in GWB
    LT.createGWB(psrtmp, Agwb, 13./3., seed=None)

    for ii in range(len(psrtmp)):
        psrtmp[ii].fit(iters=5)

    ## Save par and tim files
    for ii in range(len(psrtmp)):
        psrtmp[ii].savepar(datadir+'/{0}_optstatsim.par'.format(psrtmp[ii].name))
        psrtmp[ii].savetim(datadir+'/{0}_optstatsim.tim'.format(psrtmp[ii].name))

    return psrtmp


def run_noise_analysis(dataset, niter=1000000):

    # make the hdf5 file for the simulated dataset
    datadir = '../data/simulated_data/' + dataset
    h5filename = datadir + '/sim.hdf5'

    os.system('python makeH5file.py \
              --pardir {0} --timdir {0} \
              --h5File {1}'.format(datadir, h5filename));

    # run the common red noise analysis
    chaindir = '../data/chains/' + dataset

    os.system('PAL2_run.py --h5File {0} --pulsar all \
              --outDir {1}/fix_spec_nf_30/ --niter {2} --fixWhite \
              --noVaryNoise --mark9 --incRed --incGWB --Tspan 0 --noCorrelations --nf 30 \
              --fixSi 4.33'.format(h5filename, chaindir, niter))

    # run the individual noise analyses for each pulsar and make noise files
    pfile = h5.File(h5filename)
    psrs = list(pfile.keys())

    for psr in psrs:
        os.system('PAL2_run.py --h5File {0} --pulsar {1} \
                  --outDir chains/{2}/noise/{1}/ --niter 10000 \
                  --mark9 --incRed --nf 30 --resume'.format(h5filename, psr, dataset))

    # get noise files
    noisedir = '../data/noisefiles/' + dataset
    os.system('mkdir -p {0}'.format(noisedir))
    for psr in psrs:
        print 'Making noise files for {0}\n'.format(psr)
        cp = pp.ChainPP('chains/{0}/noise/{1}/'.format(dataset, psr))
        ml = cp.get_ml_values(mtype='marg')
        noisefile = noisedir + '/{0}_noise.txt'.format(psr)
        with open(noisefile, 'w') as f:
            for key, val in ml.items():
                f.write('%s %g\n'%(key, val))
