#!/usr/bin/env python

# run as a script with
# mpirun -np 4 pairs_mpi..py

from mpi4py import MPI
import numpy as np 
import sys
import math
import time
from scipy.spatial import cKDTree

def main():

    comm = MPI.COMM_WORLD
    size = comm.Get_size()
    rank = comm.Get_rank()
    
    print 'I am rank:', rank
    
    #read in data
    big_array = sys.argv[1]
    filename_1 = sys.argv[2]
    filename_2 = sys.argv[3]
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
    x = np.memmap(filename, mode='r+', dtype='bool_', shape=(N1,N2,N3))
    
    inds = np.arange(0,N1)
    
    sendbuf=[]
    if comm.rank==0:
        chunks = np.array_split(inds,size)
        sendbuf = chunks
        
    inds=comm.scatter(sendbuf,root=0)
    	
    #create trees
    KDT_1 = cKDTree(data_1[inds])
    KDT_2 = cKDTree(data_2)
    
    #record pairs
    for i in range(0,len(bins)):
    	pairs = np.array(KDT_1.query_ball_tree(KDT_2, bins[i]))
    	for j in range(0,len(inds)):
    		if len(pairs[j])>0:
    			x[inds[j],pairs[j],i] = True
    			if i>0:
    				x[inds[j],:,i] = x[j,:,i] - x[j,:,i-1]
    
    result='done'
    
    recvbuf=comm.gather(result,root=0)
    
    if comm.rank==0:
        print recvbuf

if __name__ == '__main__':
    main()