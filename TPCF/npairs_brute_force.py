#!/usr/bin/env python

#Duncan Campbell
#Yale University
#July 24, 2014
#calculate pair counts with dumb brute force method as a sanity check for other codes.

from __future__ import division, print_function
import numpy as np
import sys

def main():
    '''
    example:
    python npairs_brute_force.py
    '''
    import matplotlib.pyplot as plt 
    
    #define samples
    N1 = 100
    N2 = 100
    x1 = np.random.random((N1,3))
    x2 = np.random.random((N2,3))
    
    #define radial bins
    r = np.arange(0,0.5,0.1)
    
    #define PBCs
    period = np.array([1,1,1]) #periodic box with sides of length 1
    
    #number of pairs less than r
    n = npairs(x1, x2, r, period)
    
    #to get pairs in bins take diff
    DD = np.diff(n)
    
    plt.figure()
    plt.plot(r[1:], DD, 'o')
    plt.xlabel('r')
    plt.ylabel('N')
    plt.xlim([0,1])
    plt.show()


def npairs(data1, data2, r, period=None):
    '''
    Calculate the number of pairs with separations less than r.
    parameters
        data1: (n1,k) array of points
        data2: (n2,k) array of points
        r: 1-d array of radial distances to count pairs less than
        period: (k,) array defining axis-aligned periodic boundary conditions.  If none, 
            PBCs are set to infinity
    returns
        n: 
    '''
    
    #work with arrays!
    data1 = np.asarray(data1)
    data2 = np.asarray(data2)
    r = np.asarray(r)
    
    #Check to make sure both data sets have the same dimension. Otherwise, throw an error!
    if np.shape(data1)[-1]!=np.shape(data2)[-1]:
        raise ValueError("data1 and data2 inputs do not have the same dimension.")
        return None
        
    #Process period entry and check for consistency.
    if period is None:
            period = np.array([np.inf]*np.shape(data1)[-1])
    else:
        period = np.asarray(period).astype("float64")
        if np.shape(period)[0] != np.shape(data1)[-1]:
            raise ValueError("period should have len == dimension of points")
            return None
    
    N1 = len(data1)
    N2 = len(data2)
    dd = np.zeros((N1*N2,)) #store radial pair seperations 
    for i in range(0,N1): #calculate distance between every point and every other point
        x1 = data1[i,:]
        x2 = data2
        dd[i*N2:i*N2+N2] = _distance(x1, x2, period)
        
    #sort results
    dd.sort()
    #count number less than r
    n = np.zeros((len(r),))
    for i in range(len(r)): #this is ugly... is there a sexier way?
        if r[i]>np.min(period)/2:
            print("Warning: counting pairs with seperations larger than period/2 is awkward.")
            print("r=", r[i], "  min(period)/2=",np.min(period)/2)
        n[i] = len(np.where(dd<=r[i])[0])
    
    return n


def _distance(x1, x2, period=None):
    '''
    Calculate the cartesian distance between points with periodic boundary conditions.
    parameters
        x1: point or (n,k) array of points
        x2: point or (n,k) array of points
        period: (k,) array defining axis-aligned periodic boundary conditions.  If none, 
            PBCs are set to infinity
    return
        d: returns (n,) vector of distances
    '''
    
    x1 = np.asarray(x1)
    x2 = np.asarray(x2)
    
    d = np.minimum(np.fabs(x1 - x2), period - np.fabs(x1 - x2))
    d = d * d
    d = np.sqrt(np.sum(d, axis=1))
    
    return d


if __name__ == '__main__':
    main()
    
    