import numpy as np
import os,sys,glob,math
import h5py as h5

import libstempo.toasim as LTsim
import NX01_psr


# Steve's code to get pulsar properties that we want for our simulated pulsar timing data
def get_pulsar_params():
    # name, hdf5-path, par-path, tim-path
    psr_pathinfo = np.genfromtxt('../data/psrList_sim2.txt', dtype=str, skip_header=2)

    tmp_psr = []
    for ii,tmp_name in enumerate(psr_pathinfo[0:10,0],start=0):
        tmp_psr.append(h5.File(psr_pathinfo[ii,1], 'r')[tmp_name])

    psr = [NX01_psr.PsrObjFromH5(p) for p in tmp_psr]

    # Grab all the pulsar quantities
    [p.grab_all_vars() for p in psr]

    # simulation noise properties
    pwhite = np.array([1.58,2.60,1.47,1.99,1.65,0.26,0.65,1.51,0.12,1.19])
    predAmp = np.array([-13.90,-14.14,-13.09,-13.24,-18.56,-14.9,-13.6,-16.0,-13.99,-13.87])
    predGam = np.array([3.18,2.58,1.65,0.03,4.04,4.85,2.00,1.35,2.06,4.02])

    # reading in sim2 par files
    parfiles = sorted(glob.glob('../data/sim_parfiles/*_stripped.par'))

    for ii in range(len(psr)):
        print psr[ii].name, parfiles[ii], pwhite[ii], predAmp[ii], predGam[ii]

    return psr, parfiles, pwhite, predAmp, predGam


# Steve's code to create simulated pulsar timing data with specified properties
def create_dataset(dataset, Agwb):

    print 'Getting pulsar parameters for simulated dataset...'
    psr, parfiles, pwhite, predAmp, predGam = get_pulsar_params()

    datadir = '../data/simulated_data/' + dataset
    if not os.path.exists(datadir):
        os.makedirs(datadir)
    print 'Making simulated data in directory ' + datadir

    psrtmp = []
    for ii in range(len(psr)):
    
        psrtmp.append(LTsim.fakepulsar(parfile = parfiles[ii],
                                       obstimes = psr[ii].toas,
                                       toaerr = psr[ii].toaerrs/1e-6))

    ## Add in pulsar white noise

    for ii in range(len(psrtmp)):
        LTsim.add_efac(psrtmp[ii], efac=1.0)

    for ii in range(len(psrtmp)):
        psrtmp[ii].fit(iters=5)

    ## Add in red noise with same spectral properties as Sim2

    for ii in range(len(psrtmp)):
        LTsim.add_rednoise(psrtmp[ii],10.0**predAmp[ii],predGam[ii],
                        components=50,seed=None)

    for ii in range(len(psrtmp)):
        psrtmp[ii].fit(iters=5)

    ## Add in GWB
    LTsim.createGWB(psrtmp, Agwb, 13./3., seed=None)

    for ii in range(len(psrtmp)):
        psrtmp[ii].fit(iters=5)

    ## Save par and tim files
    for ii in range(len(psrtmp)):
        psrtmp[ii].savepar(datadir+'/{0}_optstatsim.par'.format(psrtmp[ii].name))
        psrtmp[ii].savetim(datadir+'/{0}_optstatsim.tim'.format(psrtmp[ii].name))

    # make the hdf5 file for the simulated dataset
    print 'Making the hdf5 file for the simulated dataset...'
    datadir = '../data/simulated_data/' + dataset
    h5filename = datadir + '/sim.hdf5'
    
    os.system('python makeH5file.py \
              --pardir {0} --timdir {0} \
              --h5File {1}'.format(datadir, h5filename));

    print 'Finished!'
