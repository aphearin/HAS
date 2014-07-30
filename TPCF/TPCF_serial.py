#!/usr/bin/env python

#Duncan Campbell
#Yale University
#June 17, 2014
#calculate the 2-point correlation function serially

from __future__ import division, print_function

def main():
    '''
    example:
    python TPCF_serial.py output.dat input1.dat input2.dat
        output.dat: valid filepath/filename to save output
        input1.dat: ascii file with N1 rows by k columns for N1 data points of dimension k
        input2.dat: ascii file with N2 rows by k columns for N2 data points of dimension k
    '''
    
    import sys
    import os
    import numpy as np
    from astropy.io import ascii
    from astropy.table import Table
    
    print("running cross-correlation...")
    if len(sys.argv)==1:
        savename = './test/test_data/test_out.dat'
        filename_1 = './test/test_data/test_D.dat'
        filename_2 = './test/test_data/test_R.dat'
        if (not os.path.isfile(filename_1)) | (not os.path.isfile(filename_2)):
            raise ValueError('Please provide vailid filenames for data input.')
        else:
            print('Running code with default test data files. Saving output as:', savename)
    elif len(sys.argv)==4:
        savename = sys.argv[1]
        filename_1 = sys.argv[2]
        filename_2 = sys.argv[3]
        print('Running code with user supplied data files. Saving output as:', savename)
    else:
        raise ValueError('Please provide a fielpath to ave output and two data files to read.')
    
    #read in data
    from astropy.io import ascii
    from astropy.table import Table
    data_1 = ascii.read(filename_1)
    data_2 = ascii.read(filename_2) 
        
    #convert tables into numpy arrays
    data_1 = np.array([data_1[c] for c in data_1.columns]).T
    data_2 = np.array([data_2[c] for c in data_2.columns]).T
    print("points in data 1: {0}  points in data 2: {1}".format(len(data_1),len(data_2)))
        
    #define log radial bins(if using PBCs, max radial bin cannot be larger than min PBC)
    bins = np.arange(-3,np.log10(0.5),0.1)
    bins = 10.0**bins
    bin_centers = (bins[:-1]+bins[1:])/2.0
    print("radial bins: {0}".format(bins))
    
    #define periodic boundary conditions
    period = np.array([1,1,1], dtype=np.float64)
    print("periodic boundary conditions: {0}".format(period))
    
    
    corr,bins = TPCF(data_1, data_2, bins, period=period)
    
    data = Table([bins[1:], corr], names=['r', 'corr'])
    ascii.write(data, savename)
    print("done.")


def TPCF(data_1, data_2, bins, period=None):
    '''
    Calculates the two point correlation function.
    paramaters
    data_1: N,k array or list of points
    data_2: N,k array or list of points
    returns
    correlation function, bins
    '''
    
    import math
    import numpy as np
    
    if np.max(bins)>np.min(period)/2:
        print(np.max(bins),np.min(period)/2)
        raise ValueError('Cannot calculate for seperations larger than min(period)')
    
    #count pairs
    DD,RR,DR,bins = npairs(data_1, data_2, bins, period)
    factor = (len(data_2)*1.0)/len(data_1)
    
    #use estimator of 2PCF
    if period==None:
        corr = TPCF_estimator(DD,RR,DR,factor)
    else: #don't need randoms if we have PBCs
        N = len(data_1)*len(data_2)
        V = np.prod(period)
        n = N/V
        corr = TPCF_estimator_periodic(DR,n,bins,period)
    
    return corr, bins


def TPCF_estimator(DD,RR,DR,factor):
    '''
    Landy-Szalay estimator for the correlation function.
    returns NaN's where RR==0.
    '''
    import numpy as np
    #dont raise an error if there is a division by 0
    old_settings = np.seterr(divide='ignore', invalid='ignore')
    
    DD.astype(np.float64)
    DR.astype(np.float64)
    RR.astype(np.float64)
    corr = (factor ** 2.0 * DD - 2.0 * factor * DR + RR) / RR
    
    np.seterr(**old_settings)
    
    return corr


def TPCF_estimator_periodic(DR,n,bins,period):
    '''
    TPCF estimator for a Euclidean manifold with periodic boundary conditions.
    '''
    import numpy as np
    
    m = len(period)
    dV = nball_volume(bins,m)
    dV = np.diff(dV)
    RR = dV*n
    
    corr = DR/RR - 1.0
    
    return corr


def nball_volume(R,m):
    import math
    
    return (math.pi**(m/2)/math.gamma(m/2+1))*R**m


def npairs(data_1, data_2, bins, period=None):
    '''
    Count pairs with separations given by bins.
    parameters
        data_1: N,k array or list of points
        data_2: N,k array or list of points
        bins: array or list of continuos bins edges
    returns:
        DD_11: data_1-data_1 pairs (auto correlation)
        DD_22: data_1-data_2 pairs (auto correlation)
        DD_12: data_2-data_2 pairs (cross-correlation)
        bins
    '''
    import numpy as np
    #from scipy.spatial import cKDTree
    from kdtrees.ckdtree import cKDTree
    
    data_1 = np.asarray(data_1)
    data_2 = np.asarray(data_2)
    bins   = np.asarray(bins)
    
    KDT_1 = cKDTree(data_1)
    KDT_2 = cKDTree(data_2)
    
    counts_11 = KDT_1.count_neighbors(KDT_1, bins, period=period)
    if data_1 != data_2:
        counts_22 = KDT_2.count_neighbors(KDT_2, bins, period=period)
        counts_12 = KDT_1.count_neighbors(KDT_2, bins, period=period)
    else:
        counts_22 = counts_11
        counts_12 = counts_11
    
    DD_11     = np.diff(counts_11)
    DD_22     = np.diff(counts_22)
    DD_12     = np.diff(counts_12)
    
    return DD_11, DD_22, DD_12, bins


if __name__ == '__main__':
    main()
    

