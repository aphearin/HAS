#!/usr/bin/env python

#Duncan Campbell
#Yale University
#July 24, 2014
#calculate pair counts with dumb brute force method as a sanity check for other codes.

from __future__ import division, print_function
import numpy as np

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


def wnpairs(data1, data2, r, period=None, weights1=None, weights2=None):
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
    if data1.ndim ==1: data1 = np.array([data1])
    data2 = np.asarray(data2)
    if data2.ndim ==1: data2 = np.array([data2])
    r = np.asarray(r)
    if r.size ==1: r = np.array([r])
    
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
    
    #Process weights1 entry and check for consistency.
    if weights1 is None:
            weights1 = np.array([1.0]*np.shape(data1)[0], dtype=np.float64)
    else:
        weights1 = np.asarray(weights1).astype("float64")
        if np.shape(weights1)[0] != np.shape(data1)[0]:
            raise ValueError("weights1 should have same len as data1")
            return None
    #Process weights2 entry and check for consistency.
    if weights2 is None:
            weights2 = np.array([1.0]*np.shape(data2)[0], dtype=np.float64)
    else:
        weights2 = np.asarray(weights2).astype("float64")
        if np.shape(weights2)[0] != np.shape(data2)[0]:
            raise ValueError("weights2 should have same len as data2")
            return None
    
    N1 = len(data1)
    N2 = len(data2)
    dd = np.zeros((N1,N2), dtype=np.float64) #store radial pair seperations 
    for i in range(0,N1): #calculate distance between every point and every other point
        x1 = data1[i,:]
        x2 = data2
        dd[i,:] = distance(x1, x2, period)
        
    #count number less than r
    n = np.zeros((r.size,), dtype=np.float64)
    for i in range(r.size): #this is ugly... is there a sexier way?
        if r[i]>np.min(period)/2:
            print("Warning: counting pairs with seperations larger than period/2 is awkward.")
            print("r=", r[i], "  min(period)/2=",np.min(period)/2)
        for j in range(N1):
            n[i] += np.sum(np.extract(dd[j,:]<=r[i],weights2))*weights1[j]

    return n


def distance(x1, x2, period=None):
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
    
    #Process period entry and check for consistency.
    if period is None:
            period = np.array([np.inf]*np.shape(x1)[-1])
    else:
        period = np.asarray(period).astype("float64")
        if np.shape(period)[0] != np.shape(x1)[-1]:
            raise ValueError("period should have len == dimension of points")
            return None
    
    x1 = np.asarray(x1)
    x2 = np.asarray(x2)
    
    d = np.minimum(np.fabs(x1 - x2), period - np.fabs(x1 - x2))
    d = d * d
    d = np.sqrt(np.sum(d, axis=len(np.shape(d))-1))
    
    return d


if __name__ == '__main__':
    main()
    
    