#!/usr/bin/env python

#Duncan Campbell
#Yale University
#July 19, 2014
#calculate all pairs and store the numbers in bins

from __future__ import print_function
from mpi4py import MPI
import numpy as np 
import sys
import os
import math
import time
from scipy.spatial import cKDTree

def main():
    '''
    example:
    mpirun -np 4 python npairs_mpi.py output.dat input1.dat input2.dat
    '''
    comm = MPI.COMM_WORLD
    rank = comm.rank
    
    #this first option is for my test files, so only works if you have a copy or make one!
    if len(sys.argv)==1:
        savename = './test/test_data/test_out.dat'
        filename_1 = './test/test_data/test_D.dat'
        filename_2 = './test/test_data/test_R.dat'
        if (not os.path.isfile(filename_1)) | (not os.path.isfile(filename_2)):
            raise ValueError('Please provide vailid filenames for data input.')
        else:
            if rank==0:
                print('Running code with default test data files. Saving output as:', savename)
    elif len(sys.argv)==4:
        savename = sys.argv[1]
        filename_1 = sys.argv[2]
        filename_2 = sys.argv[3]
        print('Running code with user supplied data files. Saving output as:', savename)
    else:
        raise ValueError('Please provide a fielpath to save output and two data files to read.')
    
    from astropy.io import ascii
    from astropy.table import Table
    data_1 = ascii.read(filename_1)
    data_2 = ascii.read(filename_2) 
        
    #convert tables into numpy arrays
    data_1 = np.array([data_1[c] for c in data_1.columns]).T
    data_2 = np.array([data_2[c] for c in data_2.columns]).T
    
    #define radial bins.  This should be made an input at some point.
    #bins = np.arange(0,1,0.01)
    #bins = np.arange(-4,1,0.1)
    bins=np.logspace(-1,1.5,10)
    bin_centers = (bins[:-1]+bins[1:])/2.0
    
    DD,RR,DR,bins = npairs(data_1, data_2, bins, period=None, comm=comm)
    
def npairs(data_1, data_2, bins, period=None, comm=None):
    '''
    Count pairs with separations given by bins.
    parameters
        data_1: N,k array or list of points
        data_2: N,k array or list of points
        bins: array or list of continuos bins edges
        period: length k array defining axis aligned PBCs. If set to none, PBCs = infinity.
        comm: mpi Intracommunicator object, or None (run on 1 core)
    returns:
        DD_11: data_1-data_1 pairs (auto correlation)
        DD_22: data_1-data_2 pairs (auto correlation)
        DD_12: data_2-data_2 pairs (cross-correlation)
        bins
    '''
    if comm==None: 
        rank = 0
        size = 1
    else: 
        rank = comm.rank
        size = comm.Get_size()
    
    N1 = len(data_1)
    N2 = len(data_2)
    N3 = len(bins)
    
    #define the indices
    inds1 = np.arange(0,N1)
    inds2 = np.arange(0,N2)
    
    #split up indices for each subprocess
    sendbuf_1=[] #need these as place holders till each process get its list
    sendbuf_2=[]
    if rank==0:
        chunks = np.array_split(inds1,size)
        sendbuf_1 = chunks
        chunks = np.array_split(inds2,size)
        sendbuf_2 = chunks
    
    if comm!=None:
        #send out lists of indices for each subprocess to use
        inds1=comm.scatter(sendbuf_1,root=0)
        inds2=comm.scatter(sendbuf_2,root=0)
    
    #creating trees seems very cheap, so I don't worry about this too much.
    #create full trees
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
    
    if comm==None:
        return DD_11, DD_22, DD_12, bins
    else:
        #gather results from each subprocess
        DD_11 = comm.gather(DD_11,root=0)
        DD_22 = comm.gather(DD_22,root=0)
        DD_12 = comm.gather(DD_12,root=0)
        
    if (rank==0) & (comm!=None):
        #combine counts from subprocesses
        DD_11=np.sum(DD_11, axis=0)
        DD_22=np.sum(DD_22, axis=0)
        DD_12=np.sum(DD_12, axis=0)
        #send result out to other processes
        DD_11 = comm.bcast(DD_11, root=0)
        DD_22 = comm.bcast(DD_22, root=0)
        DD_12 = comm.bcast(DD_12, root=0)
    
    #receive result from rank 0
    DD_11 = comm.bcast(DD_11, root=0)
    DD_22 = comm.bcast(DD_22, root=0)
    DD_12 = comm.bcast(DD_12, root=0)
    
    return DD_11, DD_22, DD_12, bins


if __name__ == '__main__':
    main()
    