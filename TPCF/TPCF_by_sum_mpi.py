#!/usr/bin/env python

#Duncan Campbell
#Yale University
#July 21, 2014
#This take an array N1 x N2 x N_bins comprised of 0's and 1's that indicates whether data
#  are pairs in the bin.  This routine does summations along the array axis to determine
#  the two point correlation function

from mpi4py import MPI
import numpy as np 
import sys
import math
import time
from astropy.io import ascii
from astropy.table import Table

def main():
    '''
    example:
    mpirun -np 4 TPCF_by_sum_mpi.py pairs_array.py
    '''
    big_array = sys.argv[1]

    comm = MPI.COMM_WORLD
    size = comm.Get_size()
    rank = comm.Get_rank()
    
    #first, a random distribution
    N1 = 1000
    N2 = 1000
    N3 = 50
    
    filename = big_array
    x = np.memmap(filename, mode='r', dtype='bool_', shape=(N1,N2,N3))
    
    inds = np.arange(0,N1)
    
    sendbuf=[]
    
    if rank==0:
        chunks = np.array_split(inds,size)
        sendbuf = chunks
        
    inds=comm.scatter(sendbuf,root=0)
    
    #print "rank:", rank, "has inds:", inds
    
    result = np.sum(x[inds,:],axis=1) #do summation on chunk of array
    
    recvbuf=comm.gather(result,root=0)
    
    if comm.rank==0:
        DD = np.concatenate(recvbuf)
        DD = np.sum(DD, axis=0)
        DD = np.diff(DD)
        print DD

    
if __name__ == '__main__':
    main()