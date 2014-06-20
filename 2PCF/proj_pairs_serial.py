#!/usr/bin/env python

import numpy as np
import matplotlib.pyplot as plt
import sys
from mpi4py import MPI
import time
from scipy.spatial import cKDTree

#Takes two data sets set_1 and set_2.
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
    bins = np.arange(-4,1,0.1)
    bins = 10.0**bins
    bin_centers = (bins[:-1]+bins[1:])/2.0
    
    #define radial bins
    bins = np.arange(-4,1,0.1)
    bins = 10.0**bins
    bin_centers = (bins[:-1]+bins[1:])/2.0
    
    N3 = len(bins)
    
    #open storage array
    filename = big_array
    x = np.memmap(filename, dtype='bool_', mode='r+', shape=(N1,N2,N3))
    
    #create tree structures
    xyz_1[:,0],xyz_1[:,1],xyz_1[:,2] = _spherical_to_cartesian(data_1[:,0], data_1[:,1])
    xyz_2[:,0],xyz_2[:,1],xyz_2[:,2] = _spherical_to_cartesian(data_2[:,0], data_2[:,1])
    KDT_1 = cKDTree(data_1)
    KDT_2 = cKDTree(data_2)

    for i in range(0,len(theta_bins)):
    	#calculate bins for angular separations
    	pairs = np.array(KDT_1.query_ball_tree(KDT_2, theta_bins[i]))
    	#convert angular separation into projected physical separation
    	r_proj = r_comov(theta_bins[i],z)
    	#calculate which r_proj bin each entry falls in
    	k_ind = np.searchsorted(r_bins,r_proj)
    	for j in range(0,N1):
    		if len(pairs[j])>0:
    			x[j,pairs[j],k_ind[j]] = True
    			if i>0: #remove previous matches
    				x[j,:,k_ind[j]] = x[j,:,k_ind[j]] - x[j,:,k_ind[j]-1]


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
    

if __name__ == '__main__':
    main()