#!/usr/bin/env python

import numpy as np
import matplotlib.pyplot as plt
import sys
from mpi4py import MPI
import time
from scipy.spatial import cKDTree

#Takes two data sets: set_1 and set_2.
# set_1 of length N1 must have ra,dec,z
# set_2 of length N2 must ave ra,dec
#returns the object pairs between set_1 and set_2
# with physical separations given by r_bins
# where to calculate the pair separation the redshift is used from the object in set_1

def main():
    
    big_array = sys.argv[1]
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
    
    N1 = len(data_1)
    N2 = len(data_2)
    
    #define angular bins(should probably be finer than the r bins)
    theta_bins = np.arange(-4,1,0.1)
    theta_bins = 10.0**bins
    theta_bin_centers = (bins[:-1]+bins[1:])/2.0
    
    #define radial bins
    r_bins = np.arange(-4,1,0.1)
    r_bins = 10.0**bins
    r_bin_centers = (bins[:-1]+bins[1:])/2.0
    
    N3 = len(bins)
    
    #open storage array
    filename = big_array

    x = proj_pairs_serial(data_1,data_2,filename)


def proj_pairs_serial(data_1,data_2,filename)
    x = np.memmap(filename, dtype='bool_', mode='r+', shape=(N1,N2,N3))
    
    #create tree structures
    xyz_1[:,0],xyz_1[:,1],xyz_1[:,2] = _spherical_to_cartesian(data_1[:,0], data_1[:,1])
    xyz_2[:,0],xyz_2[:,1],xyz_2[:,2] = _spherical_to_cartesian(data_2[:,0], data_2[:,1])
    KDT_1 = cKDTree(data_1)
    KDT_2 = cKDTree(data_2)
    
    #convert angular bins to cartesian
    c_bins = _chord_to_cartesian(theta_bins)
    
    #run a tree query for each theta bin
    for i in range(0,len(theta_bins)):
        #calculate bins for angular separations
        pairs = np.array(KDT_1.query_ball_tree(KDT_2, c_bins[i]))
        #convert angular separation into projected physical separation
        r_proj = _r_comov(theta_bins[i],z)
        #calculate which r_proj bin each entry falls in
        k_ind = np.searchsorted(r_bins,r_proj)
        for j in range(0,N1):
            if len(pairs[j])>0:
                x[j,pairs[j],k_ind[j]] = True
                if i>0: #remove previous matches
                    x[j,:,k_ind[j]] = x[j,:,k_ind[j]] - x[j,:,k_ind[j]-1]

    return x


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


def r_comov_intrpd(theta, z, cosmo='default')
    import numpy as np
    if cosmo == 'default':
        from astropy.cosmology import FlatLambdaCDM
        cosmo = FlatLambdaCDM(H0=70, Om0=0.3)
        print('warning: using default cosmolgoy in "r_comov_intrpd" func')
    
    #build interpolation function
    f = build_r_comov_intrpd(cosmo)
    #use function to calculate cmoving distance for each redshift
    X = f(z)
    
    #calculate the physical projected separation
    theta = np.radians(theta)
    d = X/(1.0+z)*theta
    
    return d


def build_r_comov_intrpd(cosmo)
    from astropy.cosmology.funcs import comoving_distance
    from scipy import interpolate
    import numpy as np

    z = np.linspace(0.0,5.0,1000)
    X = comoving_distance(z, cosmo=cosmo)
    
    f = interpolate.interp1d(z, X)
    
    return f


if __name__ == '__main__':
    main()