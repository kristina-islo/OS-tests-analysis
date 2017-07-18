import numpy as np
import os,sys,glob,math

datadir = '../data/optstat/'
datasetname = 'dataset'
ndatasets = 100

outputfilename = datadir + datasetname + 'stats.dat'

f = open(outputfilename, 'w')

for n in range(ndatasets):
    name = datadir + datasetname + str(n)
    if os.path.exists(name):
        data_init = np.loadtxt(name + '/init_os.dat', usecols=1)
        data = np.loadtxt(name + '/marg_os.dat')

        f.write('{0:<10}  {1:>10.3e}  {2:>10.3e}  {3:>6.3f}  {4:>10.3e}  {5:>10.3e}  {6:>6.3f}\n'.format(datasetname+str(n),
                                                             data_init[0], data_init[1], data_init[2],
                                                             np.mean(data[:,0]),
                                                             np.mean(data[:,1]),
                                                             np.mean(data[:,2])))

f.close()
