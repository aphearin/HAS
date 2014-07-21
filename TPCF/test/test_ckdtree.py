#!/usr/bin/env python

from __future__ import division
from ..kdtrees.ckdtree import cKDTree
import numpy as np

def test_query_ball_tree():

    data_1 = np.random.random((100,3))
    data_2 = np.random.random((100,3))
    
    tree_1 = cKDTree(data_1)
    tree_2 = cKDTree(data_2)
    
    print tree_1.query_ball_tree(tree_2,0.1)