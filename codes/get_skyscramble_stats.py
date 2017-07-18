import numpy as np
import os,sys,glob,math

datadir = '../data/optstat/'
datasetname = 'dataset'
ndatasets = 100

outputfilename = datadir + datasetname + 'stats_skyscrambles.dat'

f = open(outputfilename, 'w')

for n in range(ndatasets):
    name = datadir + datasetname + str(n)
    if os.path.exists(name):
        data = np.loadtxt(name + '/marg_os.dat')
        snr = np.mean(data[:,2])
        
        nscrambles = 210
        scrambled_snr = []
        for N in range(nscrambles):
            ss = np.loadtxt(name + '/marg_os_skyscramble{0}.dat'.format(N))
            scrambled_snr.append(np.mean(ss[:,2]))
        
        count = 0
        for i in range(len(scrambled_snr)):
            if scrambled_snr[i] >= snr:
                count += 1
    
        pvalue = float(count)/float(nscrambles)

        f.write('{0:<10}  {1:>3.0f}  {2:.4f}\n'.format(datasetname+str(n), count, pvalue))

f.close()
