#!/usr/bin/env python

from __future__ import division
from ..weighted_npairs_brute_force import wnpairs
import numpy as np


def test_npairs():
    N=100
    x1 = np.random.random((N,3))
    weights2 = np.zeros(N)+0.1
    
    n = wnpairs(x1,x1,np.sqrt(3.0),weights2=weights2)
    
    print n[0], N*N*0.1
    assert np.around(n[0],decimals=4)==np.around(N*N*0.1,decimals=4), "not all pairs found/weights wrong"

def test_npairs_periodic():
    N=100
    x1 = np.random.random((N,3))
    weights2 = np.zeros(N)+0.1
    period = np.array([1,1,1])
    
    n = wnpairs(x1,x1,np.min(period)/2.0,period=period, weights2=weights2)
    
    #not sure of a good test here...
    assert n[0]<N*N, 'this would be bad...'
    
    