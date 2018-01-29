import numpy as np
import sys, time


def compute_match(orf1, orf2):
    
    orf1_mag = np.sqrt(np.dot(orf1,orf1))
    orf2_mag = np.sqrt(np.dot(orf2,orf2))

    match = np.abs(np.dot(orf1, orf2))/(orf1_mag*orf2_mag)

    if match > 1.0:
        print 'Match is > 1.0! What is going on?'
        print orf1
        print orf2

    return match


# returns the Hellings and Downs coefficient for two DIFFERENT pulsars
#    x :   cosine of the angle between the pulsars
def HD(x):
    return 1.5*(1./3. + (1.-x)/2.*(np.log((1.-x)/2.)-1./6.))


# computes the number of pulsars based on the number of pulsar pairs
def compute_npsrs(npairs):

    n = (1 + np.sqrt(1+8*npairs))/2

    return int(n)


def compute_orf(ptheta, pphi):
    
    npsr = len(ptheta)
    pos = [ np.array([np.cos(phi)*np.sin(theta),
                      np.sin(phi)*np.sin(theta),
                      np.cos(theta)]) for phi, theta in zip(pphi, ptheta) ]
                      
    x = []
    for i in range(npsr):
        for j in range(i+1,npsr):
            x.append(np.dot(pos[i], pos[j]))
    x = np.array(x)

    orf = HD(x)

    return orf


def get_scrambles(orf_true, N=500, Nmax=10000, thresh=0.2, save=False, filename='../data/scrambles_nano_thresh02.npy'):
    """
    Get sky scramble ORFs and matches.
    
    :param orf_true: The true ORF values (vector of off diagonal elements)
    :param N: Number of sky scrambles
    :param Nmax: Maximum number of tries to get independent scrambles
    :param thresh: Threshold value for match statistic.
    """

    print 'Generating {0} sky scrambles from {1} attempts with threshold {2}...'.format(N, Nmax, thresh)
    npsr = compute_npsrs(len(orf_true))
    matchs, orfs = [], []
    ct = 0
    tstart = time.time()
    while ct <= Nmax and len(matchs) <= N:
        ptheta = np.arccos(np.random.uniform(-1, 1, npsr))
        pphi = np.random.uniform(0, 2*np.pi, npsr)
        orf_s = compute_orf(ptheta, pphi)
        match = compute_match(orf_true, orf_s)
        if match <= thresh:
            if len(orfs) == 0:
                orfs.append(orf_s)
                matchs.append(match)
            else:
                check = np.all(map(lambda x: compute_match(orf_s, x)<=thresh, orfs))
                if check:
                    orfs.append(orf_s)
                    matchs.append(match)

        ct += 1
        if ct % 1000 == 0:
            sys.stdout.write('\r')
            sys.stdout.write('Finished %2.1f percent in %f min'
                                 % (float(ct)/Nmax*100, (time.time() - tstart)/60. ))
            sys.stdout.flush()

    if len(matchs) < N:
        print '\nGenerated {0} matches rather than the desired {1} matches'.format(len(matchs), N)
    else:
        print '\nGenerated the desired {0} matches in {1} attempts'.format(len(matchs), ct)
    print 'Total runtime: {0:.1f} min'.format((time.time()-tstart)/60.)

    if save:
        np.save(filename, orfs)

    return (matchs, orfs)


if __name__ == '__main__':

    import argparse

    parser = argparse.ArgumentParser(description='')

    parser.add_argument('--orf_true', default='../data/orf_nano.npy')
    parser.add_argument('--threshold', default=0.2, help='threshold for sky scrambles (DEFAULT 0.2)')
    parser.add_argument('--nscrambles', default=1000, help='number of sky scrambles to generate (DEFAULT 1000)')
    parser.add_argument('--nmax', default=1000, help='maximum number of attempts (DEFAULT 1000)')
    parser.add_argument('--savefile', default='../data/scrambles_nano.npy', help='outputfile name')

    args = parser.parse_args()

    orf0 = np.load(args.orf_true)

    get_scrambles(orf0, N=int(args.nscrambles), Nmax=int(args.nmax), thresh=float(args.threshold), save=True, filename=args.savefile)
