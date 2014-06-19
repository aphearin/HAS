#!/usr/bin/env python

import numpy as np
import matplotlib.pyplot as plt
import sys
from mpi4py import MPI
import time
from scipy.spatial import cKDTree

def main():

    filename = sys.argv[1]

    N1 = int(sys.argv[2])
    N2 = int(sys.argv[3])
    N3 = int(sys.argv[4])
    
    x = np.memmap(filename, dtype='bool_', mode='w+', shape=(N1,N2,N3))
    
    print x.shape, x.nbytes/1073741824.0    
    
if __name__ == '__main__':
    main()