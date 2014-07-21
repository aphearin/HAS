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

def main():

    big_array = sys.argv[1]
    N1 = 1000
    N2 = 1000
    N3 = 50
    
    filename = big_array
    x = np.memmap(filename, mode='r', dtype='bool_', shape=(N1,N2,N3))
    
    DD = np.sum(x,axis=1)
    DD = np.sum(DD, axis=0)
    #DD = np.diff(DD)
    
    print DD
    
if __name__ == '__main__':
    main()