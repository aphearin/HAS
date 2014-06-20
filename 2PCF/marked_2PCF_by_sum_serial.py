#!/usr/bin/env python

# run as a script with
# mpirun -np 4 helloworld.py

from mpi4py import MPI
import numpy as np 
import sys
import math
import time

def main():

    big_array = sys.argv[1]
    N1 = 1000
    N2 = 1000
    N3 = 50
    
    filename = big_array
    x = np.memmap(filename, mode='r', dtype='bool_', shape=(N1,N2,N3))
    
    mark_1 = np.arange(0,N1)
    mark_2 = np.arange(0,N2)
    
    #if marks are axis independent
    DD = np.sum(x[mark_1],axis=1)
    DD = np.sum(DD[mark_2], axis=0)
    #DD = np.diff(DD)
    
    #if marks are axis dependent option 1
    #create 2-d mask
    import numpy.ma as ma
    mark12 = ma.make_mask_none((N1,N2))==False
    
    DD = np.sum(x[mark12,:],axis=1)
    DD = np.sum(DD, axis=0)
    
    #if marks are axis dependent option 2
    #rank ordered objects on axis and sum over index range
    ind1 = np.arange(0,N1)
    low_mark = np.zeros(N1)
    upper_mark = f_ind2(ind1)
    
    DD = np.zeros((N1,N3))
    for i in range(0,N1):
    	DD[i,:] = np.sum(x[i,low_mark[i]:upper_mark[i],:],axis=1)
    DD = np.sum(DD, axis=0)
    
    print DD
    
if __name__ == '__main__':
    main()