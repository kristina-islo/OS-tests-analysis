# modification of simulation code that injects a common signal
# with power-law PSD and dipole spatial correlations

from __future__ import division

import numpy as np
import os,sys,glob,math
import json

import scipy.interpolate as interp

import ephem
import libstempo.toasim as LT
from libstempo import spharmORFbasis as anis


def createdipole(psr, Amp, gam, noCorr=False, seed=None, turnover=False,
                 clm=[np.sqrt(4.0*np.pi)], lmax=0, f0=1e-9, beta=1,
                 power=1, userSpec=None, npts=600, howml=10):
    """
        Function to create GW-induced residuals from a dipole background as defined
        in Chamberlin, Creighton, Demorest, et al. (2014)
        with modification to the ORF by SJV
        
        :param psr: pulsar object for single pulsar
        :param Amp: Amplitude of red noise in GW units
        :param gam: Red noise power law spectral index
        :param noCorr: Add red noise with no spatial correlations
        :param seed: Random number seed
        :param turnover: Produce spectrum with turnover at frequency f0
        :param clm: coefficients of spherical harmonic decomposition of GW power
        :param lmax: maximum multipole of GW power decomposition
        :param f0: Frequency of spectrum turnover
        :param beta: Spectral index of power spectram for f << f0
        :param power: Fudge factor for flatness of spectrum turnover
        :param userSpec: User-supplied characteristic strain spectrum
        (first column is freqs, second is spectrum)
        :param npts: Number of points used in interpolation
        :param howml: Lowest frequency is 1/(howml * T)
        
        :returns: list of residuals for each pulsar
        """
    
    if seed is not None:
        np.random.seed(seed)
    
    # number of pulsars
    Npulsars = len(psr)

    # gw start and end times for entire data set
    start = np.min([p.toas().min()*86400 for p in psr]) - 86400
    stop = np.max([p.toas().max()*86400 for p in psr]) + 86400
    
    # duration of the signal
    dur = stop - start
    
    # get maximum number of points
    if npts is None:
        # default to cadence of 2 weeks
        npts = dur/(86400*14)

    # make a vector of evenly sampled data points
    ut = np.linspace(start, stop, npts)

    # time resolution in days
    dt = dur/npts
    
    # compute the overlap reduction function
    psrlocs = np.zeros((Npulsars,2))
    
    for ii in range(Npulsars):
        if 'RAJ' and 'DECJ' in psr[ii].pars():
            psrlocs[ii] = np.double(psr[ii]['RAJ'].val), np.double(psr[ii]['DECJ'].val)
        elif 'ELONG' and 'ELAT' in psr[ii].pars():
            fac = 180./np.pi
            # check for B name
            if 'B' in psr[ii].name:
                epoch = '1950'
            else:
                epoch = '2000'
            coords = ephem.Equatorial(ephem.Ecliptic(str(psr[ii]['ELONG'].val*fac),
                                                     str(psr[ii]['ELAT'].val*fac)),
                                      epoch=epoch)
            psrlocs[ii] = float(repr(coords.ra)), float(repr(coords.dec))

    psrlocs[:,1] = np.pi/2. - psrlocs[:,1]
        
    ORF = np.zeros((Npulsars,Npulsars))
    for i in range(Npulsars):
        for j in range(Npulsars):
#            zeta = anis.calczeta(psrlocs[:,0][i],psrlocs[:,0][j],
#                                 psrlocs[:,1][i],psrlocs[:,1][j])
#            ORF[i,j] = np.cos(zeta)
            if i == j:
                # linear algebra wizardry courtesy of Steve
                # this ensures the matrix is positive definite so the Cholesky transform can be used
                ORF[i,j] = 1. + 1e-6
            else:
                ORF[i,j] = np.sin(psrlocs[i,1])*np.sin(psrlocs[j,1])*np.cos(psrlocs[i,0]-psrlocs[j,0]) + \
                            np.cos(psrlocs[i,1])*np.cos(psrlocs[j,1])
    ORF *= 2.0
   
    # Define frequencies spanning from DC to Nyquist.
    # This is a vector spanning these frequencies in increments of 1/(dur*howml).
    f = np.arange(0, 1/(2*dt), 1/(dur*howml))
    f[0] = f[1] # avoid divide by 0 warning
    Nf = len(f)

    # Use Cholesky transform to take 'square root' of ORF
    M = np.linalg.cholesky(ORF)
    
    # Create random frequency series from zero mean, unit variance, Gaussian distributions
    w = np.zeros((Npulsars, Nf), complex)
    for ll in range(Npulsars):
        w[ll,:] = np.random.randn(Nf) + 1j*np.random.randn(Nf)
    
    # strain amplitude
    if userSpec is None:
        
        f1yr = 1/3.16e7
        alpha = -0.5 * (gam-3)
        hcf = Amp * (f/f1yr)**(alpha)
        if turnover:
            si = alpha - beta
            hcf /= (1+(f/f0)**(power*si))**(1/power)

    elif userSpec is not None:
    
        freqs = userSpec[:,0]
        if len(userSpec[:,0]) != len(freqs):
            raise ValueError("Number of supplied spectral points does not match number of frequencies!")
        else:
            fspec_in = interp.interp1d(np.log10(freqs), np.log10(userSpec[:,1]), kind='linear')
            fspec_ex = extrap1d(fspec_in)
            hcf = 10.0**fspec_ex(np.log10(f))

    C = 1 / 96 / np.pi**2 * hcf**2 / f**3 * dur * howml
    
    ### injection residuals in the frequency domain
    Res_f = np.dot(M, w)
    for ll in range(Npulsars):
        Res_f[ll] = Res_f[ll] * C**(0.5)    # rescale by frequency dependent factor
        Res_f[ll,0] = 0                # set DC bin to zero to avoid infinities
        Res_f[ll,-1] = 0            # set Nyquist bin to zero also

    # Now fill in bins after Nyquist (for fft data packing) and take inverse FT
    Res_f2 = np.zeros((Npulsars, 2*Nf-2), complex)
    Res_t = np.zeros((Npulsars, 2*Nf-2))
    Res_f2[:,0:Nf] = Res_f[:,0:Nf]
    Res_f2[:, Nf:(2*Nf-2)] = np.conj(Res_f[:,(Nf-2):0:-1])
    Res_t = np.real(np.fft.ifft(Res_f2)/dt)

    # shorten data and interpolate onto TOAs
    Res = np.zeros((Npulsars, npts))
    res_gw = []
    for ll in range(Npulsars):
        Res[ll,:] = Res_t[ll, 10:(npts+10)]
        f = interp.interp1d(ut, Res[ll,:], kind='linear')
        res_gw.append(f(psr[ll].toas()*86400))
    
    #return res_gw
    ct = 0
    for p in psr:
        p.stoas[:] += res_gw[ct]/86400.0
        ct += 1


def create_dataset(dataset, Agwb):
    """ Create simulated dataset using 18 pulsars from NANOGrav 9-year
        stochastic analysis. Will use 11-year data span and red noise values
        with white noise values taken from most recent time-to-detection
        simulations.
        :param dataset: Name of output dataset.
        :param Agwb: Amplitude of injected dipole signal
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

    # dipole signal
    createdipole(psrs, Agwb, 13./3., seed=None)
    
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
