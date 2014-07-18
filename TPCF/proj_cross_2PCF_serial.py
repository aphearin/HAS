#!/usr/bin/env python

from __future__ import division

def main():
    '''
    Example code to calculate the projected 2PCF using proj_cross_npairs_serial().
    '''
    import sys
    import matplotlib.pyplot as plt
    import numpy as np
    
    #filename_1 = sys.argv[1]
    #filename_2 = sys.argv[2]
    #filename_3 = sys.argv[3]
    filename_1 = 'test/test_radec_1.dat'
    filename_2 = 'test/test_radec_2.dat'
    filename_3 = 'test/test_ran_radec_2.dat'
    
    #read in data
    from astropy.io import ascii
    from astropy.table import Table
    data_1 = ascii.read(filename_1)
    data_2 = ascii.read(filename_2) 
    ran_2 = ascii.read(filename_3)
        
    #convert tables into numpy arrays
    data_1 = np.array([data_1[c] for c in data_1.columns]).T
    data_2 = np.array([data_2[c] for c in data_2.columns]).T
    ran_2 = np.array([ran_2[c] for c in ran_2.columns]).T
    
    N1 = len(data_1)
    N2 = len(data_2)
    N3 = len(ran_2)
    
    #define radial bins
    r_bins = np.arange(-3,1,0.2) #Mpc
    r_bins = 10.0**r_bins
    r_bin_centers = (r_bins[:-1]+r_bins[1:])/2.0
    
    #define cosmology
    from astropy.cosmology import FlatLambdaCDM
    cosmo = FlatLambdaCDM(H0=70, Om0=0.3)

    #count pairs
    DD = proj_cross_npairs_serial(data_1,data_2,r_bins,cosmo)
    print DD
    DR = proj_cross_npairs_serial(data_1,ran_2,r_bins,cosmo)
    print DR
    
    factorDD = len(data_1)/len(data_2)
    factorDR = len(data_2)/len(ran_2)
    
    cross_proj_corr = TPCF_estimator(DD,DR,factorDR)
    print cross_proj_corr
    
    plt.figure()
    plt.plot(r_bin_centers,cross_proj_corr[:-1] + 1)
    plt.xlabel('R Mpc')
    plt.ylabel(r'$\xi+1$')
    plt.show()


'''
def TPCF_estimator(DD,RR,DR,factor):
    #estimate correlation function
    corr = (factor ** 2.0 * DD - 2.0 * factor * DR + RR) / RR

    return corr
'''
    
    
def TPCF_estimator(DD,RR,factor):
    #estimate correlation function
    corr = (1.0/factor)*DD/RR - 1.0

    return corr


def proj_cross_npairs_serial(data_1,data_2,r_bins,cosmo):
    '''
    parameters
        data_1: ra, dec, z
        data_2: ra, dec
        r_bins: physical radial bins in kpc
        cosmo: astropy cosmology object
    returns
        N pairs in radial bins
    '''
    from astropy.cosmology.funcs import comoving_distance
    import numpy as np
    #from scipy.spatial import cKDTree
    from ckdtree import cKDTree #modified version of scipy.spatial.ckdtree code
    
    #create tree structures for angular pair calculation
    xyz_1 = np.empty((len(data_1),3))
    xyz_2 = np.empty((len(data_2),3))
    xyz_1[:,0],xyz_1[:,1],xyz_1[:,2] = _spherical_to_cartesian(data_1[:,0], data_1[:,1])
    xyz_2[:,0],xyz_2[:,1],xyz_2[:,2] = _spherical_to_cartesian(data_2[:,0], data_2[:,1])
    
    KDT_1 = cKDTree(xyz_1)
    KDT_2 = cKDTree(xyz_2)
    
    #redshifts of sample 1
    z = data_1[:,2]
    #comoving distance to sample 1
    X = comoving_distance(z, cosmo=cosmo).value
    
    #define angular bins given r_proj bins and redshift range
    theta_bins = _proj_r_to_angular_bins(r_bins, z, 10, cosmo)
    
    #convert angular bins to cartesian distances
    c_bins = _chord_to_cartesian(theta_bins)
    
    #run a tree query for each theta bin
    pp = np.zeros(len(r_bins)) #pair count storage array
    N1 = len(data_1)
    prev_pairs = np.zeros(len(data_1))
    for i in range(0,len(theta_bins)):
        #calculate bins for angular separations
        pairs = np.array(KDT_1.query_ball_tree(KDT_2, c_bins[i]))
        #convert angular separation into projected physical separation
        r_proj = X/(1.0+z)*theta_bins[i]
        #calculate which r_proj bin each theta_bin falls in
        k_ind = np.searchsorted(r_bins,r_proj)
        for j in range(0,N1):
            if k_ind[j]<len(pp):
                pp[k_ind[j]] += pairs[j] - prev_pairs[j]
        prev_pairs = pairs

    return pp


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


def _proj_r_to_angular_bins(r_bins, z, N_sample, cosmo):
    '''
    define angular bins given r_proj bins and redshift range.
    parameters
        r_bins: np.array, projected radial bins in Mpc
        N_sample: int, oversample rate of theta bins
        cosmo: astropy cosmology object defining cosmology
    returns:
        theta_bins: np.array, angular bins in radians
    '''
    import numpy as np
    from astropy.cosmology.funcs import comoving_distance
    
    N_r_bins = len(r_bins)
    N_theta_bins = N_sample * N_r_bins
    
    #find maximum theta
    X_min = comoving_distance(np.min(z),cosmo).value
    max_theta = np.max(r_bins)/(X_min/(1.0+np.min(z)))
    
    #find minimum theta
    X_max = comoving_distance(np.max(z),cosmo).value
    min_theta = np.min(r_bins)/(X_max/(1.0+np.max(z)))
    
    theta_bins = np.linspace(np.log10(min_theta), np.log10(max_theta), N_theta_bins)
    theta_bins = 10.0**theta_bins
    
    return theta_bins


#def build_comov_intrpd(cosmo):
#    from astropy.cosmology.funcs import comoving_distance
#    from scipy import interpolate
#    import numpy as np
#
#    z = np.linspace(0.0,1.0,1000)
#    X = comoving_distance(z, cosmo=cosmo)
#    
#    f = interpolate.interp1d(z, X, kind='linear') #need to look into speed of various options
#    
#    return f


if __name__ == '__main__':
    main()