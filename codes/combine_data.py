# Collates optimal statistic data into one file containing 
# both types of fixed noise and noise-marginalized results 


import numpy as np
import os,sys,glob,math

datadir = '../data/optstat/'
datasetname = 'dataset'
ndatasets = 100

outputfilename = datadir + 'stats_all.dat'

f = open(outputfilename, 'w')

for n in range(ndatasets):
    name = '../data/optstat_individual/' + dataset + str(n)
    if os.path.exists(name):
        data_init_individual = np.loadtxt(name + '/init_os.dat', usecols=1)
        data_individual = np.loadtxt(name + '/marg_os.dat')
    name = '../data/optstat_common/' + dataset + str(n)
    if os.path.exists(name):
        data_init_common = np.loadtxt(name + '/init_os.dat', usecols=1)
        data_common = np.loadtxt(name + '/marg_os.dat')

        f.write('{0:<10}  {1:>10.3e}  {2:>10.3e}  {3:>6.3f}  {4:>10.3e}  {5:>10.3e}  {6:>6.3f} {7:>10.3e}  {8:>10.3e}  {9:>6.3f}  {10:>10.3e}  {11:>10.3e}  {12:>6.3f}\n'.format(datasetname+str(n),
                                                             data_init_individual[0], data_init_individual[1], data_init_individual[2],
                                                             np.mean(data_individual[:,0]),
                                                             np.mean(data_individual[:,1]),
                                                             np.mean(data_individual[:,2]), 
                                                             data_init_common[0], data_init_common[1], data_init_common[2],
                                                             np.mean(data_common[:,0]),
                                                             np.mean(data_common[:,1]),
                                                             np.mean(data_common[:,2])))
f.close()
