#!/usr/bin/env python

#Duncan Campbell
#Yale University
#June 17, 2014
#calculate the 2-point correlation function serially


def main():
    '''
    example:
    python angular_TPCF_serial.py output.dat input1.dat input2.dat
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
    bins = np.arange(-3,1,0.1)
    bins = 10.0**bins
    bin_centers = (bins[:-1]+bins[1:])/2.0
    
    #count pairs
    DD,RR,DR,bins = npairs(data_1, data_2, bins)
    factor = (len(data_2)*1.0)/len(data_1)
    
    #use estimator of 2PCF
    corr = TPCF_estimator(DD,RR,DR,factor)
    
    data = Table([bins[1:], corr], names=['theta', 'corr'])
    ascii.write(data, savename)
    print "done."


def angular_TPCF(data_1, data_2, bins):
    '''
    Calculates the angular two point correlation function.
    paramaters
    data_1: N,k array or list of ra,dec coordinates in degrees
    data_2: N,k array or list of ra,dec coordinates in degrees
    bins: array defining angular bin edges in degrees.  bins must be continuous
    returns
    correlation function, bins
    '''
    
    xyz_1 = np.empty((len(data_1),3))
    xyz_2 = np.empty((len(data_2),3))
    xyz_1[:,0],xyz_1[:,1],xyz_1[:,2] = _spherical_to_cartesian(data_1[:,0], data_1[:,1])
    xyz_2[:,0],xyz_2[:,1],xyz_2[:,2] = _spherical_to_cartesian(data_2[:,0], data_2[:,1])
    
    #convert angular bins to cartesian distances
    c_bins = _chord_to_cartesian(bins, radians=False)
    
    #count pairs
    DD,RR,DR,bins = npairs(xyz_1, xyz_2, c_bins)
    factor = (len(data_2)*1.0)/len(data_1)
    
    #use estimator of 2PCF
    corr = TPCF_estimator(DD,RR,DR,factor)
    
    return corr, bins


def TPCF_estimator(DD,RR,DR,factor):
    '''
    Landy-Szalay estimater for the correlation function
    '''
    corr = (factor ** 2.0 * DD - 2.0 * factor * DR + RR) / RR
    
    return corr
    
    
def npairs(data_1, data_2, bins):
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


def _spherical_to_cartesian(ra, dec):
    '''
    Calculate cartesian coordinates on a unit sphere given two angular coordinates. 
    parameters
        ra: np.array of angular coordinate in degrees
        dec: np.array of angular coordinate in degrees
    returns
        x,y,z: np.arrays cartesian coordinates 
    '''
    from numpy import radians, sin, cos
    
    rar = radians(ra)
    decr = radians(dec)
    
    x = cos(rar) * cos(decr)
    y = sin(rar) * cos(decr)
    z = sin(decr)
    
    return x, y, z


def _chord_to_cartesian(theta, radians=True):
    '''
    Calculate chord distance on a unit sphere given an angular distance between two 
    points.
    parameters
        theta: np.array of angular distance
        radians: input in radians.  Default is true  If False, input in degrees.
    returns
        C: np.array chord distance 
    '''
    from numpy import radians, sin
    if radians==False: theta = radians(theta)
    
    C = 2.0*sin(theta/2.0)
    
    return C


if __name__ == '__main__':
    main()
    

