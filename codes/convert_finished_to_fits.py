import numpy as np
import os.path
import glob
from astropy.io import fits
from astropy.table import Table


def chain_txt_to_fits(chaindir):
    
    txtfile = '{0}/chain_1.txt'.format(chaindir)
    parfile = '{0}/pars.txt'.format(chaindir)
    
    chain = np.loadtxt(txtfile)
    pars = np.loadtxt(parfile, dtype='S42')
    
    # Append pars with MCMC information, these are always the 
    # last four entries in a chain sample
    super_pars = np.array(['lnprob', 'lnlike', 'acceptance rate', 'parallel-tempering acceptance rate'])

    pars = np.concatenate((pars, super_pars))
    
    # initiate array of of columns
    c = [[]] * len(pars)

    # fill in columns with chain data
    for ii, par in enumerate(pars):
        c[ii] = fits.Column(name=par, format='D', array=chain[:,ii]) # format is set to float64: 'D'

    # convert collection of columns to Bin HDU table
    tbhdu = fits.BinTableHDU.from_columns(c)

    # save as fits file
    print 'Writing fits file {0}/chain.fits'.format(chaindir)
    tbhdu.writeto('{0}/chain.fits'.format(chaindir))

    print 'Removing chain file ' + txtfile
    os.remove(txtfile)
    

if __name__ == '__main__':

    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--chaindir')

    args = parser.parse_args()

    chain_txt_to_fits(chaindir=args.chaindir)
