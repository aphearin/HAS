#!/usr/bin/env python

from __future__ import division
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
    
    #define radial bins
    r_bins = np.arange(-4,1,0.1)
    r_bins = 10.0**r_bins
    r_bin_centers = (r_bins[:-1]+r_bins[1:])/2.0
    
    #define cosmology
    from astropy.cosmology import FlatLambdaCDM
    cosmo = FlatLambdaCDM(H0=70, Om0=0.3)

    DD = proj_cross_npairs_serial(data_1,data_2,r_bins,cosmo)
    DR = proj_cross_npairs_serial(data_1,ran_2,r_bins,cosmo)
    RR = proj_cross_npairs_serial(ran_1,ran_2,r_bins,cosmo)
    
    factorDD = len(data_1)/len(data_2)
    factorDR = len(data_1)/len(ran_2)
    factorRR = len(ran1)/len(ran2)
    
    cross_proj_corr = TPCF_estimator(DD,RR,DR,factor)


def TPCF_estimator(DD,RR,DR,factor):
    #estimate correlation function
    corr = (factor ** 2.0 * DD - 2.0 * factor * DR + RR) / RR

    return corr


def proj_cross_npairs_serial(data_1,data_2,r_bins,cosmo)
    '''
    parameters
        data_1: ra, dec, z
        data_2: ra, dec
        r_bins: physical radial bins in kpc
        cosmo: astropy cosmology object
    returns
        N pairs in radial bins
    '''
    
    #create tree structures for angular pair calculation
    xyz_1[:,0],xyz_1[:,1],xyz_1[:,2] = _spherical_to_cartesian(data_1[:,0], data_1[:,1])
    xyz_2[:,0],xyz_2[:,1],xyz_2[:,2] = _spherical_to_cartesian(data_2[:,0], data_2[:,1])
    KDT_1 = cKDTree(data_1)
    KDT_2 = cKDTree(data_2)
    
    #redshifts of sample 1
    z = data_1[:,2]
    
    #build redshift angular distance interpolation function
    comov = build_comov_intrpd(cosmo)
    
    #define angular bins given r_proj bins and redshift range
    N_r_bins = len(r_bins)
    N_theta_bins = 10 * N_r_bins
    max_theta = np.max(r_bins)/(X/(1.0+np.min(z)))
    min_theta = np.min(r_bins)/(X/(1.0+np.max(z)))
    theta_bins = np.linspace(np.log10(min_theta), np.log10(max_theta), N_theta_bins)
    theta_bins = 10.0**theta_bins
    
    #convert angular bins to cartesian
    c_bins = _chord_to_cartesian(theta_bins)
    
    #create array to stor number counts
    pp = np.zeros(len(r_bins)-1)
    
    #run a tree query for each theta bin
    for i in range(0,len(theta_bins)):
        #calculate bins for angular separations
        pairs = np.array(KDT_1.query_ball_tree(KDT_2, c_bins[i]))
        #convert angular separation into projected physical separation
        X = comov(z)
        r_proj = X/(1.0+z)*theta[i]
        #calculate which r_proj bin each entry falls in
        k_ind = np.searchsorted(r_bins,r_proj)
        for j in range(0,N1):
            pp += len(pairs[j])
    
    pp = np.diff(pp)

    return pp


def _spherical_to_cartesian(ra, dec):

    from numpy import radians, sin, cos

    rar = radians(ra)
    decr = radians(dec)

    x = cos(rar) * cos(decr)
    y = sin(rar) * cos(decr)
    z = sin(decr)
 
    return x, y, z              


def _chord_to_cartesian(C):
    
    from numpy import radians, arcsin
    
    C = radians(C)

    d = 2.0*arcsin(C/2.0)
    
    return d


def build_comov_intrpd(cosmo)
    from astropy.cosmology.funcs import comoving_distance
    from scipy import interpolate
    import numpy as np

    z = np.linspace(0.0,5.0,1000)
    X = comoving_distance(z, cosmo=cosmo)
    
    f = interpolate.interp1d(z, X)
    
    return f


if __name__ == '__main__':
    main()