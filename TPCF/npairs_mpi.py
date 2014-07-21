#!/usr/bin/env python

#Duncan Campbell
#Yale University
#July 19, 2014
#calculate all pairs and store the numbers in bins

from mpi4py import MPI
import numpy as np 
import sys
import math
import time
from scipy.spatial import cKDTree

def main():
    '''
    example:
    mpirun -np 4 python npairs_mpi.py output.dat input1.dat input2.dat
    '''
    
    comm = MPI.COMM_WORLD
    size = comm.Get_size()
    rank = comm.Get_rank()
    
    print 'running rank:', rank
    
    #read in data
    filename_1 = sys.argv[2]
    filename_2 = sys.argv[3]
    from astropy.io import ascii
    from astropy.table import Table
    data_1 = ascii.read(filename_1)
    data_2 = ascii.read(filename_2) 
        
    #convert tables into numpy arrays
    data_1 = np.array([data_1[c] for c in data_1.columns]).T
    data_2 = np.array([data_2[c] for c in data_2.columns]).T
    
    #how many points are in each array
    N1 = len(data_1)
    N2 = len(data_2)
    if comm.rank==0:
        print N1, 'x', N2, 'points'
    
    #define radial bins.  This should be made an input at some point.
    bins = np.arange(0,1,0.01)
    bins = np.arange(-4,1,0.1)
    bins = 10.0**bins
    bin_centers = (bins[:-1]+bins[1:])/2.0
    
    N3 = len(bins)
    
    #define the indices
    inds1 = np.arange(0,N1)
    inds2 = np.arange(0,N1)
    
    #split up indices for each subprocess
    sendbuf_1=[] #need these as place holders till each process get its list
    sendbuf_2=[]
    if comm.rank==0:
        chunks = np.array_split(inds1,size)
        sendbuf_1 = chunks
        chunks = np.array_split(inds2,size)
        sendbuf_2 = chunks
    
    #send out lists of indices for each subprocess to use
    inds1=comm.scatter(sendbuf_1,root=0)
    inds2=comm.scatter(sendbuf_2,root=0)
    
    #create trees
    KDT_1 = cKDTree(data_1)
    KDT_2 = cKDTree(data_2)
    #create chunked up trees
    KDT_1_small = cKDTree(data_1[inds1])
    KDT_2_small = cKDTree(data_2[inds2])
    
    #count!
    counts_11 = KDT_1_small.count_neighbors(KDT_1, bins)
    counts_22 = KDT_2_small.count_neighbors(KDT_2, bins)
    counts_12 = KDT_1_small.count_neighbors(KDT_2, bins)
    DD_11     = np.diff(counts_11)
    DD_22     = np.diff(counts_22)
    DD_12     = np.diff(counts_12)
    
    #gather results from each subprocess
    DD_11 = comm.gather(DD_11,root=0)
    DD_22 = comm.gather(DD_22,root=0)
    DD_12 = comm.gather(DD_12,root=0)
    
    if comm.rank==0:
        #combine counts from subprocesses
        DD_11=np.sum(DD_11, axis=0)
        DD_22=np.sum(DD_22, axis=0)
        DD_12=np.sum(DD_12, axis=0)
        savename= sys.argv[1]
        data = Table([bins[1:], DD_11, DD_22, DD_12], names=['r', 'DD_11','DD_22','DD_12'])
        ascii.write(data, savename)

if __name__ == '__main__':
    main()
    