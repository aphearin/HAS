#!/usr/bin/env python

from __future__ import division
from ..kdtrees.ckdtree import cKDTree
from ..npairs_brute_force import npairs
from ..pairs_brute_force import pairs
from ..weighted_npairs_brute_force import wnpairs
import numpy as np
import sys

def test_count_neighbors():

    data_1 = np.random.random((100,3))
    data_2 = np.random.random((100,3))
    
    tree_1 = cKDTree(data_1)
    tree_2 = cKDTree(data_2)
    
    n1 = npairs(data_1,data_2,0.25)[0]
    
    n2 = tree_1.count_neighbors(tree_2,0.25)
    
    assert n1==n2, 'tree calc did not find same number of pairs'


def test_count_neighbors_periodic():

    data_1 = np.random.random((100,3))
    data_2 = np.random.random((100,3))
    
    tree_1 = cKDTree(data_1)
    tree_2 = cKDTree(data_2)
    
    period = np.array([1,1,1])
    
    n1 = npairs(data_1,data_2,0.25, period=period)[0]
    
    n2 = tree_1.count_neighbors(tree_2,0.25, period=period)
    
    assert n1==n2, 'tree calc did not find same number of pairs'


def test_query_pairs():
    data_1 = np.random.random((100,3))
    
    tree_1 = cKDTree(data_1)
    
    p1 = pairs(data_1, data_1, 0.25)
    p2 = tree_1.query_pairs(0.25)
    
    print(len(p1), len(p2))
    p3 = p1.intersection(p2)
    assert (len(p2)==len(p3)) & (len(p2)==(len(p1)-len(data_1))/2), 'not all pairs found'


def test_query_pairs_periodic():
    data_1 = np.random.random((100,3))
    
    period = np.array([1,1,1])
    
    tree_1 = cKDTree(data_1)
    
    p1 = pairs(data_1, data_1, 0.25, period=period)
    p2 = tree_1.query_pairs(0.25, period=period)
    
    print(len(p1), len(p2))
    p3 = p1.intersection(p2)
    assert (len(p2)==len(p3)) & (len(p2)==(len(p1)-len(data_1))/2), 'not all pairs found'


def test_query_ball_tree():
    data_1 = np.random.random((100,3))
    tree_1 = cKDTree(data_1)
    data_2 = np.random.random((100,3))
    tree_2 = cKDTree(data_2)
    
    n1 = npairs(data_1, data_2, 0.4)
    n2 = tree_1.query_ball_tree(tree_2, 0.4)
    n2 = sum(len(x) for x in n2) #number of pairs found
    
    assert n1==n2, 'inconsistent number found'


def test_query_ball_tree_periodic():
    data_1 = np.random.random((100,3))
    tree_1 = cKDTree(data_1)
    data_2 = np.random.random((100,3))
    tree_2 = cKDTree(data_2)
    
    period = np.array([1,1,1])
    
    n1 = npairs(data_1, data_2, 0.4, period=period)
    n2 = tree_1.query_ball_tree(tree_2, 0.4, period=period)
    n2 = sum(len(x) for x in n2) #number of pairs found
    
    assert n1==n2, 'inconsistent number found'


def test_query_ball_point():
    data_1 = np.random.random((100,3))
    tree_1 = cKDTree(data_1)
    
    #define point
    x = np.array([0.5,0.5,0.5])
    
    n1 = npairs(data_1,x,0.5)[0]
    n2 = len(tree_1.query_ball_point(x,0.5))
    
    assert n1==n2, 'inconsistent number of points found...'


def test_query_ball_point_periodic():
    data_1 = np.random.random((100,3))
    tree_1 = cKDTree(data_1)
    
    #define point
    x = np.array([0.1,0.1,0.1])
    
    period = np.array([1,1,1])
    
    n1 = npairs(data_1,x,0.4, period=period)[0]
    n2 = len(tree_1.query_ball_point(x,0.4, period=period))
    
    assert n1==n2, 'inconsistent number of points found...'


def test_query(): #not a very good test...
    data_1 = np.random.random((100,3))
    tree_1 = cKDTree(data_1)
    
    #define point
    x = np.array([0.5,0.5,0.5])
    
    ps = tree_1.query(x,10)[0]
    print ps
    assert len(ps)==10, 'inconsistent number of points found...'


def test_query_periodic(): #not a very good test...
    data_1 = np.random.random((100,3))
    tree_1 = cKDTree(data_1)
    
    #define point
    x = np.array([0.5,0.5,0.5])
    
    period = np.array([1,1,1])
    
    ps = tree_1.query(x,10,period=period)[0]
    print ps
    assert len(ps)==10, 'inconsistent number of points found...'


def test_wcount_neighbors_periodic():
    #need to know the float precision of the computer
    epsilon = np.float64(sys.float_info[8])

    N1 = 100
    N2 = 100
    data_1 = np.random.random((N1,3))
    data_2 = np.random.random((N2,3))
    
    tree_1 = cKDTree(data_1)
    tree_2 = cKDTree(data_2)
    
    period = np.array([1,1,1])
    weights = np.zeros((N2,))+0.1
    
    n0 = wnpairs(data_1, data_2, 0.1, period=period, weights2=weights)[0]
    n1 = tree_1.wcount_neighbors(tree_2,0.1, period=period) #no weights
    n2 = tree_1.wcount_neighbors(tree_2,0.1, period=period, oweights=weights) #constant weights
    n3 = tree_1.count_neighbors(tree_2,0.1, period=period) #non-weighted function
    
    #what is the expected precision?
    ep = epsilon*np.sqrt(np.float64(N1*N2))
    
    print('brute force result:{0:0.10f} ckdtree result:{1:0.10f}'.format(n0,n2))
    assert np.fabs(n2-n3*0.1)/n0 < 10.0 * ep, 'error in weighted counts.'
    assert np.fabs(n0-n2)/n0 < 10.0 * ep, 'error in weighted.'
    
    #define random weights for test data set 2
    weights = np.random.random((len(data_2),))
    
    n0 = wnpairs(data_1, data_2, 0.25, period=period, weights2=weights)[0]
    n2 = tree_1.wcount_neighbors(tree_2,0.25, period=period, oweights=weights) #constant weights
    
    print('brute force result:{0:0.10f} ckdtree result:{1:0.10f}'.format(n0,n2))
    print('error:{0} expected error:{1}'.format(np.fabs(n0-n2)/n0,ep))
    assert np.fabs(n0-n2)/n0 < 10.0 * ep, 'weights are being handeled incorrectly'
    

def test_wcount_neighbors_large():
    return 0 #skip this as it takes some time!
    #need to know the float precision of the computer
    epsilon = np.float64(sys.float_info[8])

    #create random coordinates
    N1 = 10000
    N2 = 10000
    data_1 = np.random.random((N1,3))
    data_2 = np.random.random((N2,3))
    
    #build trees for points
    tree_1 = cKDTree(data_1)
    tree_2 = cKDTree(data_2)
    
    #define random weights for test data set 2
    weights = np.random.random((N2,))
    
    #calculate weighted sums
    n0 = wnpairs(data_1, data_2, 1.0, weights2=weights)[0]
    n1 = tree_1.wcount_neighbors(tree_2,1.0, oweights=weights)
    
    #what is the expected precision?
    ep = epsilon*np.sqrt(np.float64(N1*N2))
    
    print('brute force result:{0:0.10f} ckdtree result:{1:0.10f}'.format(n0,n1))
    print('error:{0} expected error:{1}'.format(np.fabs(n0-n1)/n0,ep))
    assert np.fabs(n0-n1)/n0 < 10.0 * ep, 'weights are being handeled incorrectly'


def test_wcount_neighbors_double_weights():
    #need to know the float precision of the computer
    epsilon = np.float64(sys.float_info[8])

    #create random coordinates
    N1 = 100
    N2 = 100
    data_1 = np.random.random((N1,3))
    data_2 = np.random.random((N2,3))
    
    #build trees for points
    tree_1 = cKDTree(data_1)
    tree_2 = cKDTree(data_2)
    
    #define random weights for test data set 2
    weights1 = np.random.random((N1,))
    weights2 = np.random.random((N2,))
    
    #calculate weighted sums
    n0 = wnpairs(data_1, data_2, 1.0, weights1=weights1, weights2=weights2)[0]
    n1 = tree_1.wcount_neighbors(tree_2,1.0, sweights=weights1, oweights=weights2)
    
    #what is the expected precision?
    ep = epsilon*np.sqrt(np.float64(N1*N2))
    
    print('brute force result:{0:0.10f} ckdtree result:{1:0.10f}'.format(n0,n1))
    print('error:{0} expected error:{1}'.format(np.fabs(n0-n1)/n0,ep))
    assert np.fabs(n0-n1)/n0 < 10.0 * ep, 'weights are being handeled incorrectly'


def test_wcount_neighbors_double_weights_functionality():
    #need to know the float precision of the computer
    epsilon = np.float64(sys.float_info[8])
    
    #user defined function
    from ..kdtrees.ckdtree import Function
    class MyFunction(Function):
        def evaluate(self, x, y, a, b):
            return x*y

    #create random coordinates
    N1 = 100
    N2 = 100
    data_1 = np.random.random((N1,3))
    data_2 = np.random.random((N2,3))
    
    #build trees for points
    tree_1 = cKDTree(data_1)
    tree_2 = cKDTree(data_2)
    
    #define random weights for test data set 2
    weights1 = np.random.random((N1,))
    weights2 = np.random.random((N2,))
    
    #calculate weighted sums
    n0 = wnpairs(data_1, data_2, 1.0, weights1=weights1, weights2=weights2)[0]
    n1 = tree_1.wcount_neighbors(tree_2,1.0, sweights=weights1, oweights=weights2, w=MyFunction())
    
    #what is the expected precision?
    ep = epsilon*np.sqrt(np.float64(N1*N2))
    
    print('brute force result:{0:0.10f} ckdtree result:{1:0.10f}'.format(n0,n1))
    print('error:{0} expected error:{1}'.format(np.fabs(n0-n1)/n0,ep))
    assert np.fabs(n0-n1)/n0 < 10.0 * ep, 'weights are being handeled incorrectly'


def test_query_ball_point_wcounts():
    #need to know the float precision of the computer
    epsilon = np.float64(sys.float_info[8])

    #create random coordinates
    N1 = 1000
    data_1 = np.random.random((N1,3))
    p=np.random.random((3,))
    
    #build trees for points
    tree_1 = cKDTree(data_1)
    
    #define random weights for test data set 2
    weights1 = np.random.random((N1,))
    
    #calculate weighted sums
    n0 = wnpairs(data_1, p, 1.0, weights1=weights1)[0]
    n1 = tree_1.query_ball_point_wcounts(p, 1.0, weights=weights1)
    
    #what is the expected precision?
    ep = epsilon*np.sqrt(np.float64(N1))
    
    print('brute force result:{0:0.10f} ckdtree result:{1:0.10f}'.format(n0,n1))
    print('error:{0} expected error:{1}'.format(np.fabs(n0-n1)/n0,ep))
    assert np.fabs(n0-n1)/n0 < 10.0 * ep, 'weights are being handeled incorrectly'
    
    



