#!/usr/bin/env python

from __future__ import division
from ..kdtrees.ckdtree import cKDTree
from ..npairs_brute_force import npairs
from ..pairs_brute_force import pairs
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


def test_query():
    pass


