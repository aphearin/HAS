#!/usr/bin/env python

import numpy as np
import matplotlib.pyplot as plt
import sys
from mpi4py import MPI
import time
from scipy.spatial import cKDTree

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
    
    bins = np.arange(0,1,0.01)
    bins = np.arange(-4,1,0.1)
    bins = 10.0**bins
    bin_centers = (bins[:-1]+bins[1:])/2.0
    
    N3 = len(bins)
    
    filename = big_array
    x = np.memmap(filename, dtype='bool_', mode='r+', shape=(N1,N2,N3))
    
    KDT_1 = cKDTree(data_1)
    KDT_2 = cKDTree(data_2)


    for i in range(0,len(bins)):
    	pairs = np.array(KDT_1.query_ball_tree(KDT_2, bins[i]))
    	for j in range(0,N1):
    		if len(pairs[j])>0:
    			x[j,pairs[j],i] = True
    			
    
if __name__ == '__main__':
    main()