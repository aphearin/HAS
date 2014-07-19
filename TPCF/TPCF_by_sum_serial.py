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
    
    DD = np.sum(x,axis=1)
    DD = np.sum(DD, axis=0)
    #DD = np.diff(DD)
    
    print DD
    
if __name__ == '__main__':
    main()