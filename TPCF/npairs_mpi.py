#!/usr/bin/env python

#Duncan Campbell
#Yale University
#July 19, 2014
#calculate all pairs and store in a large array wich can be used to calculate the TPCF by 
#  sum.  The large array should already exist.

from mpi4py import MPI
import numpy as np 
import sys
import math
import time
from scipy.spatial import cKDTree

def main():
    '''
    example:
    mpirun -np 4 python pairs_serial.py output.dat input1.dat input2.dat
    '''
    
    comm = MPI.COMM_WORLD
    size = comm.Get_size()
    rank = comm.Get_rank()
    
    print 'running rank:', rank
    
    #read in data
    filename_1 = sys.argv[1]
    filename_2 = sys.argv[2]
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
    
    inds = np.arange(0,N1)
    
    sendbuf=[]
    if comm.rank==0:
        chunks = np.array_split(inds,size)
        sendbuf = chunks
    
    inds=comm.scatter(sendbuf,root=0)
    
    #create trees
    KDT_1 = cKDTree(data_1[inds])
    KDT_2 = cKDTree(data_2)
    
    counts_11 = KDT_1.count_neighbors(KDT_1, bins)
    counts_22 = KDT_2.count_neighbors(KDT_2, bins)
    counts_12 = KDT_1.count_neighbors(KDT_2, bins)
    DD_11     = np.diff(counts_11)
    DD_22     = np.diff(counts_22)
    DD_12     = np.diff(counts_12)
    
    DD_11=comm.gather(DD_11,root=0)
    DD_22=comm.gather(DD_22,root=0)
    DD_12=comm.gather(DD_12,root=0)
    
    if comm.rank==0:
        DD_11=np.sum(DD_11, axis=0)
        DD_22=np.sum(DD_22, axis=0)
        DD_12=np.sum(DD_12, axis=0)

if __name__ == '__main__':
    main()
    