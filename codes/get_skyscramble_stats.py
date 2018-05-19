import numpy as np


def get_skyscramble_stats(datasetname, nscrambles):
    
    output = open('../data/optstat_common/{0}/skyscramble_summary.dat'.format(datasetname), 'w')

    for n in range(nscrambles):
        ss = np.loadtxt('../data/optstat_common/{0}/marg_os_skyscramble{1}.dat'.format(datasetname,n))
        output.write('skyscramble{0:<3}  {1:>18}  {2}\n'.format(n, np.mean(ss[:,0]), np.mean(ss[:,2])))

    output.close()


for i in range(299,300):
    print 'Making skyscramble summary file for dataset_nano_A1e-15_{0}'.format(i)
    get_skyscramble_stats('dataset_nano_A1e-15_{0}'.format(i), 725)
