#!/usr/bin/env python

#Duncan Campbell
#Yale University
#June 17, 2014
#calculate the 2-point correlation function serially


def main():
    '''
    example:
    python TPCF_serial.py output.dat input1.dat input2.dat
    '''
    import sys
    import numpy as np
    
    print "running cross-correlation..."
    savename = sys.argv[1]
    filename_1 = sys.argv[2]
    filename_2 = sys.argv[3]
    #read in data
    from astropy.io import ascii
    from astropy.table import Table
    data_1 = ascii.read(filename_1)
    data_2 = ascii.read(filename_2) 
        
    #convert tables into numpy arrays
    data_1 = np.array([data_1[c] for c in data_1.columns]).T
    data_2 = np.array([data_2[c] for c in data_2.columns]).T
        
    #define log radial bins
    bins = np.arange(0,1,0.01)
    bins = np.arange(-3,0.5,0.1)
    bins = 10.0**bins
    bin_centers = (bins[:-1]+bins[1:])/2.0
    
    #count pairs
    DD,RR,DR,bins = npairs(data_1, data_2, bins)
    factor = (len(data_2)*1.0)/len(data_1)
    
    print DR
    
    #use estimator of 2PCF
    corr = TPCF_estimator(DD,RR,DR,factor)
    
    data = Table([bins[1:], corr], names=['r', 'corr'])
    ascii.write(data, savename)
    print "done."


def TPCF(data_1, data_2, bins):
    '''
    Calculates the two point correlation function.
    paramaters
    data_1: N,k array or list of points
    data_2: N,k array or list of points
    returns
    correlation function, bins
    '''
    
    #count pairs
    DD,RR,DR,bins = npairs(data_1, data_2, bins)
    factor = (len(data_2)*1.0)/len(data_1)
    
    #use estimator of 2PCF
    corr = TPCF_estimator(DD,RR,DR,factor)
    
    return corr, bins


def TPCF_estimator(DD,RR,DR,factor):
    #estimate correlation function
    corr = (factor ** 2.0 * DD - 2.0 * factor * DR + RR) / RR
    
    return corr
    
    
def npairs(data_1, data_2, bins):
    #count pairs with separations given by bins
    import numpy as np
    from scipy.spatial import cKDTree
    
    data_1 = np.asarray(data_1)
    data_2 = np.asarray(data_2)
    bins   = np.asarray(bins)
    
    KDT_1 = cKDTree(data_1)
    KDT_2 = cKDTree(data_2)
    
    counts_11 = KDT_1.count_neighbors(KDT_1, bins)
    counts_22 = KDT_2.count_neighbors(KDT_2, bins)
    counts_12 = KDT_1.count_neighbors(KDT_2, bins)
    DD_11     = np.diff(counts_11)
    DD_22     = np.diff(counts_22)
    DD_12     = np.diff(counts_12)
    
    return DD_11, DD_22, DD_12, bins


if __name__ == '__main__':
    main()
    

