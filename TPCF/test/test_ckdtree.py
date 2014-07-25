#!/usr/bin/env python

from __future__ import division
from ..kdtrees.ckdtree import cKDTree
from ..npairs_brute_force import npairs
from ..pairs_brute_force import pairs
from ..weighted_npairs_brute_force import wnpairs
import numpy as np

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

    data_1 = np.random.random((100,3))
    data_2 = np.random.random((100,3))
    
    tree_1 = cKDTree(data_1)
    tree_2 = cKDTree(data_2)
    
    period = np.array([1,1,1])
    weights = np.zeros((len(data_2),))+0.1
    
    n0 = wnpairs(data_1, data_2, 0.1, period=period, weights=weights)[0]
    n1 = tree_1.wcount_neighbors(tree_2,0.1, period=period) #no weights
    n2 = tree_1.wcount_neighbors(tree_2,0.1, period=period, weights=weights) #constant weights
    n3 = tree_1.count_neighbors(tree_2,0.1, period=period) #non-weighted function
    
    print(np.around(n0, decimals=4),np.around(n2, decimals=4))
    assert np.around(n2, decimals=4)==np.around(n3*0.1, decimals=4), 'weight counts are wrong'
    assert np.around(n1, decimals=4)==np.around(n3*1.0, decimals=4), 'total counts are wrong... maybe weird weight counts still'
    assert np.around(n0, decimals=4)==np.around(n2, decimals=4), 'total counts are wrong... maybe weird weight counts still'
    
    weights = np.random.random((len(data_2),))
    
    n0 = wnpairs(data_1, data_2, 0.25, period=period, weights=weights)[0]
    n2 = tree_1.wcount_neighbors(tree_2,0.25, period=period, weights=weights) #constant weights
    
    '''
    for i in range(0,len(weights)):
        print(i,weights[i])
    '''
    
    print(np.around(n0, decimals=4),np.around(n2, decimals=4)) 
    assert np.around(n0, decimals=4)==np.around(n2, decimals=4), 'weights are being handeled incorrectly'
    

def test_wcount_neighbors_large():

    data_1 = np.random.random((10000,3))
    data_2 = np.random.random((10000,3))
    
    tree_1 = cKDTree(data_1)
    tree_2 = cKDTree(data_2)
    
    weights = np.random.random((len(data_2),))
    
    n0 = wnpairs(data_1, data_2, 1.0, weights=weights)[0]
    n2 = tree_1.wcount_neighbors(tree_2,1.0, weights=weights) #constant weights
    
    assert np.around(n0, decimals=4)==np.around(n2, decimals=4), 'weights are being handeled incorrectly'
    
    



