import numpy as np
import os,sys,glob,math
import json

import libstempo.toasim as LT

def create_dataset(dataset, Agwb):
    """ Create simulated dataset using 18 pulsars from NANOGrav 9-year
        stochastic analysis. Will use 11-year data span and red noise values
        with white noise values taken from most recent time-to-detection
        simulations.
        :param dataset: Name of output dataset.
        :param Agwb: Amplitude of injected GWB
        """
    
    print 'Getting pulsar parameters for simulated dataset...'
    # get simulation data
    with open('nano9_simdata.json', 'r') as fp:
        pdict = json.load(fp)
    
    # get red noise dictionary
    with open('nano_red_dict.json', 'r') as fp:
        red_dict = json.load(fp)

    # get parfiles
    parfiles = glob.glob('../data/nano9_stipped_parfiles/*.par')
    parfiles = [p for p in parfiles if p.split('/')[-1].split('_')[0]
                in pdict.keys()]

    datadir = '../data/simulated_data/' + dataset
    if not os.path.exists(datadir):
        os.makedirs(datadir)
    print 'Making simulated data in directory ' + datadir
    
    psrs = []
    for pf in parfiles:
        pname = pf.split('/')[-1].split('_')[0]
        psrs.append(LT.fakepulsar(pf, pdict[pname][0], pdict[pname][1]))

    for psr in psrs:
        # white noise
        LT.add_efac(psr)
        
        # red noise
        if pname in red_dict:
            LT.add_rednoise(psr, red_dict[pname][0], red_dict[pname][1],
                            components=30)

    # GWB
    LT.createGWB(psrs, Agwb, 13./3., seed=None)

    for psr in psrs:
        psr.fit(iters=2)
    
        ## Save par and tim files
        psr.savepar(datadir+'/{0}_optstatsim.par'.format(psr.name))
        psr.savetim(datadir+'/{0}_optstatsim.tim'.format(psr.name))

    # make the hdf5 file for the simulated dataset
    print 'Making the hdf5 file for the simulated dataset...'
    datadir = '../data/simulated_data/' + dataset
    h5filename = datadir + '/sim.hdf5'

    os.system('python makeH5file.py \
              --pardir {0} --timdir {0} \
              --h5File {1}'.format(datadir, h5filename));

    print 'Finished!'
