#!/usr/bin/env python

from __future__ import division
from ..TPCF_serial import TPCF
from ..TPCF_serial import npairs
import numpy as np


def test_TPCF():
    import numpy as np
    
    #Create a random 3-D distribution
    N_points_1 = 100
    N_points_2 = 20 * N_points_1 #random sample
    data_1 = np.random.random((N_points_1,3)) #random points in range {[0,1),[0,1),[0,1)}
    data_2 = np.random.random((N_points_2,3)) #random points in range {[0,1),[0,1),[0,1)}
    
    #define log radial bins
    bins = np.arange(-3,0.5,0.1)
    bins = 10.0**bins
    bin_centers = (bins[:-1]+bins[1:])/2.0
    
    xi, bins = TPCF(data_1, data_2, bins)
    
    assert len(xi)==(len(bins)-1), "result arrays are wrong length"
    
    
def test_pair_counts():
    import numpy as np
    
    #Create a random 3-D distribution
    N_points_1 = 100
    N_points_2 = 20 * N_points_1 #random sample
    data_1 = np.random.random((N_points_1,3)) #random points in range {[0,1),[0,1),[0,1)}
    data_2 = np.random.random((N_points_2,3)) #random points in range {[0,1),[0,1),[0,1)}
    
    #define log radial bins
    bins = np.arange(0,2,0.1)
    bin_centers = (bins[:-1]+bins[1:])/2.0
    
    DD_11, DD_22, DD_12, bins = npairs(data_1, data_2, bins)
    
    assert np.sum(DD_11)==len(data_1)**2-len(data_1), "not all pairs found"
    assert np.sum(DD_22)==len(data_2)**2-len(data_2), "not all pairs found"
    assert np.sum(DD_12)==len(data_1)*len(data_2), "not all pairs found"
    assert len(DD_11)==len(bins)-1, "output not right length"
    assert len(DD_12)==len(bins)-1, "output not right length"
    assert len(DD_22)==len(bins)-1, "output not right length"
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    