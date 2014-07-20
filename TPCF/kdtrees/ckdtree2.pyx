#Duncan Campbell 
#July 17, 2014
#Yale University
#subclass of scipy.spatial.cKDTree
#includes modified code from cKDTree


from scipy.spatial import cKDTree
import numpy as np
import scipy.sparse

cimport numpy as np
cimport libc.stdlib as stdlib
cimport cython

cdef extern from "limits.h":
    long LONG_MAX
cdef np.float64_t infinity = np.inf

__all__ = ['cKDTree2']

cdef inline int set_add_pair(set results,
                             np.intp_t i,
                             np.intp_t j) except -1:

    if sizeof(long) < sizeof(np.intp_t):
        # Win 64
        results.add((int(i), int(j)))
    else:
        # Other platforms
        results.add((i, j))
    return 0


cdef inline int set_add_ordered_pair(set results,
                                     np.intp_t i,
                                     np.intp_t j) except -1:

    if sizeof(long) < sizeof(np.intp_t):
        # Win 64
        if i < j:
            results.add((int(i), int(j)))
        else:
            results.add((int(j), int(i)))
    else:
        # Other platforms
        if i < j:
            results.add((i, j))
        else:
            results.add((j, i))
    return 0

cdef inline int list_append(list results, np.intp_t i) except -1:
    if sizeof(long) < sizeof(np.intp_t):
        # Win 64
        if i <= <np.intp_t>LONG_MAX:  # CHECK COMPARISON DIRECTION
            results.append(int(i))
        else:
            results.append(i)
    else:
        # Other platforms
        results.append(i)
    return 0
    


# Priority queue
# ==============
cdef union heapcontents:    # FIXME: Unions are not always portable, verify this 
    np.intp_t intdata     # union is never used in an ABI dependent way.
    char* ptrdata

cdef struct heapitem:
    np.float64_t priority
    heapcontents contents

cdef class heap(object):
    cdef np.intp_t n
    cdef heapitem* heap
    cdef np.intp_t space
    
    def __init__(heap self, np.intp_t initial_size):
        cdef void *tmp
        self.space = initial_size
        self.heap = <heapitem*> NULL
        tmp = stdlib.malloc(sizeof(heapitem)*self.space)
        if tmp == NULL:
            raise MemoryError
        self.heap = <heapitem*> tmp  
        self.n = 0

    def __dealloc__(heap self):
        if self.heap != <heapitem*> NULL:
            stdlib.free(self.heap)

    cdef inline int _resize(heap self, np.intp_t new_space) except -1:
        cdef void *tmp
        if new_space < self.n:
            raise ValueError("Heap containing %d items cannot be resized to %d" % (int(self.n), int(new_space)))
        self.space = new_space
        tmp = stdlib.realloc(<void*>self.heap, new_space*sizeof(heapitem))
        if tmp == NULL:
            raise MemoryError
        self.heap = <heapitem*> tmp
        return 0

    @cython.cdivision(True)
    cdef inline int push(heap self, heapitem item) except -1:
        cdef np.intp_t i
        cdef heapitem t

        self.n += 1
        if self.n > self.space:
            self._resize(2 * self.space + 1)
            
        i = self.n - 1
        self.heap[i] = item
        
        while i > 0 and self.heap[i].priority < self.heap[(i - 1) // 2].priority:
            t = self.heap[(i - 1) // 2]
            self.heap[(i - 1) // 2] = self.heap[i]
            self.heap[i] = t
            i = (i - 1) // 2
        return 0
    
    
    cdef heapitem peek(heap self):
        return self.heap[0]
    
    
    @cython.cdivision(True)
    cdef int remove(heap self) except -1:
        cdef heapitem t
        cdef np.intp_t i, j, k, l
    
        self.heap[0] = self.heap[self.n-1]
        self.n -= 1
        # No point in freeing up space as the heap empties.
        # The whole heap gets deallocated at the end of any query below
        #if self.n < self.space//4 and self.space>40: #FIXME: magic number
        #    self._resize(self.space // 2 + 1)
        i=0
        j=1
        k=2
        while ((j<self.n and 
                    self.heap[i].priority > self.heap[j].priority or
                k<self.n and 
                    self.heap[i].priority > self.heap[k].priority)):
            if k<self.n and self.heap[j].priority>self.heap[k].priority:
                l = k
            else:
                l = j
            t = self.heap[l]
            self.heap[l] = self.heap[i]
            self.heap[i] = t
            i = l
            j = 2*i+1
            k = 2*i+2
        return 0
    
    cdef int pop(heap self, heapitem *it) except -1:
        it[0] = self.peek()
        self.remove()
        return 0


# Utility functions
# =================
cdef inline np.float64_t dmax(np.float64_t x, np.float64_t y):
    if x>y:
        return x
    else:
        return y
        
cdef inline np.float64_t dabs(np.float64_t x):
    if x>0:
        return x
    else:
        return -x
        
# Utility for building a coo matrix incrementally
cdef class coo_entries:
    cdef:
        np.intp_t n, n_max
        np.ndarray i, j
        np.ndarray v
        np.intp_t *i_data
        np.intp_t *j_data
        np.float64_t *v_data
    
    def __init__(self):
        self.n = 0
        self.n_max = 10
        self.i = np.empty(self.n_max, dtype=np.intp)
        self.j = np.empty(self.n_max, dtype=np.intp)
        self.v = np.empty(self.n_max, dtype=np.float64)
        self.i_data = <np.intp_t *>np.PyArray_DATA(self.i)
        self.j_data = <np.intp_t *>np.PyArray_DATA(self.j)
        self.v_data = <np.float64_t*>np.PyArray_DATA(self.v)

    cdef void add(coo_entries self, np.intp_t i, np.intp_t j, np.float64_t v):
        cdef np.intp_t k
        if self.n == self.n_max:
            self.n_max *= 2
            self.i.resize(self.n_max)
            self.j.resize(self.n_max)
            self.v.resize(self.n_max)
            self.i_data = <np.intp_t *>np.PyArray_DATA(self.i)
            self.j_data = <np.intp_t *>np.PyArray_DATA(self.j)
            self.v_data = <np.float64_t*>np.PyArray_DATA(self.v)
        k = self.n
        self.i_data[k] = i
        self.j_data[k] = j
        self.v_data[k] = v
        self.n += 1

    def to_matrix(coo_entries self, shape=None):
        # Shrink arrays to size
        self.i.resize(self.n)
        self.j.resize(self.n)
        self.v.resize(self.n)
        self.i_data = <np.intp_t *>np.PyArray_DATA(self.i)
        self.j_data = <np.intp_t *>np.PyArray_DATA(self.j)
        self.v_data = <np.float64_t*>np.PyArray_DATA(self.v)
        self.n_max = self.n
        return scipy.sparse.coo_matrix((self.v, (self.i, self.j)),
                                       shape=shape)


# Measuring distances
# ===================
cdef inline np.float64_t _distance_p(np.float64_t *x, np.float64_t *y,
                                     np.float64_t p, np.intp_t k,
                                     np.float64_t upperbound):
    """Compute the distance between x and y

    Computes the Minkowski p-distance to the power p between two points.
    If the distance**p is larger than upperbound, then any number larger
    than upperbound may be returned (the calculation is truncated).
    """
    cdef np.intp_t i
    cdef np.float64_t r, z
    r = 0
    if p==2:
        for i in range(k):
            z = x[i] - y[i]
            r += z*z
            if r>upperbound:
                return r 
    elif p==infinity:
        for i in range(k):
            r = dmax(r,dabs(x[i]-y[i]))
            if r>upperbound:
                return r
    elif p==1:
        for i in range(k):
            r += dabs(x[i]-y[i])
            if r>upperbound:
                return r
    else:
        for i in range(k):
            r += dabs(x[i]-y[i])**p
            if r>upperbound:
                return r
    return r


# Interval arithmetic
# ===================

cdef class Rectangle:
    cdef np.intp_t m
    cdef np.float64_t *mins
    cdef np.float64_t *maxes
    cdef np.ndarray mins_arr, maxes_arr

    def __init__(self, mins_arr, maxes_arr):
        # Copy array data
        self.mins_arr = np.array(mins_arr, dtype=np.float64, order='C')
        self.maxes_arr = np.array(maxes_arr, dtype=np.float64, order='C')
        self.mins = <np.float64_t*>np.PyArray_DATA(self.mins_arr)
        self.maxes = <np.float64_t*>np.PyArray_DATA(self.maxes_arr)
        self.m = self.mins_arr.shape[0]

# 1-d pieces
# These should only be used if p != infinity
cdef inline np.float64_t min_dist_point_interval_p(np.float64_t* x,
                                                   Rectangle rect,
                                                   np.intp_t k,
                                                   np.float64_t p):    
    """Compute the minimum distance along dimension k between x and
    a point in the hyperrectangle.
    """
    return dmax(0, dmax(rect.mins[k] - x[k], x[k] - rect.maxes[k])) ** p

cdef inline np.float64_t max_dist_point_interval_p(np.float64_t* x,
                                                   Rectangle rect,
                                                   np.intp_t k,
                                                   np.float64_t p):
    """Compute the maximum distance along dimension k between x and
    a point in the hyperrectangle.
    """
    return dmax(rect.maxes[k] - x[k], x[k] - rect.mins[k]) ** p

cdef inline np.float64_t min_dist_interval_interval_p(Rectangle rect1,
                                                      Rectangle rect2,
                                                      np.intp_t k,
                                                      np.float64_t p):
    """Compute the minimum distance along dimension k between points in
    two hyperrectangles.
    """
    return dmax(0, dmax(rect1.mins[k] - rect2.maxes[k],
                        rect2.mins[k] - rect1.maxes[k])) ** p

cdef inline np.float64_t max_dist_interval_interval_p(Rectangle rect1,
                                                      Rectangle rect2,
                                                      np.intp_t k,
                                                      np.float64_t p):
    """Compute the maximum distance along dimension k between points in
    two hyperrectangles.
    """
    return dmax(rect1.maxes[k] - rect2.mins[k], rect2.maxes[k] - rect1.mins[k]) ** p

# Interval arithmetic in m-D
# ==========================

# These should be used only for p == infinity
cdef inline np.float64_t min_dist_point_rect_p_inf(np.float64_t* x,
                                                   Rectangle rect):
    """Compute the minimum distance between x and the given hyperrectangle."""
    cdef np.intp_t i
    cdef np.float64_t min_dist = 0.
    for i in range(rect.m):
        min_dist = dmax(min_dist, dmax(rect.mins[i]-x[i], x[i]-rect.maxes[i]))
    return min_dist

cdef inline np.float64_t max_dist_point_rect_p_inf(np.float64_t* x,
                                                   Rectangle rect):
    """Compute the maximum distance between x and the given hyperrectangle."""
    cdef np.intp_t i
    cdef np.float64_t max_dist = 0.
    for i in range(rect.m):
        max_dist = dmax(max_dist, dmax(rect.maxes[i]-x[i], x[i]-rect.mins[i]))
    return max_dist

cdef inline np.float64_t min_dist_rect_rect_p_inf(Rectangle rect1,
                                                  Rectangle rect2):
    """Compute the minimum distance between points in two hyperrectangles."""
    cdef np.intp_t i
    cdef np.float64_t min_dist = 0.
    for i in range(rect1.m):
        min_dist = dmax(min_dist, dmax(rect1.mins[i] - rect2.maxes[i],
                                       rect2.mins[i] - rect1.maxes[i]))
    return min_dist

cdef inline np.float64_t max_dist_rect_rect_p_inf(Rectangle rect1,
                                                  Rectangle rect2):
    """Compute the maximum distance between points in two hyperrectangles."""
    cdef np.intp_t i
    cdef np.float64_t max_dist = 0.
    for i in range(rect1.m):
        max_dist = dmax(max_dist, dmax(rect1.maxes[i] - rect2.mins[i],
                                       rect2.maxes[i] - rect1.mins[i]))
    return max_dist

# Rectangle-to-rectangle distance tracker
# =======================================
#
# The logical unit that repeats over and over is to keep track of the
# maximum and minimum distances between points in two hyperrectangles
# as these rectangles are successively split.
#
# Example
# -------
# # node1 encloses points in rect1, node2 encloses those in rect2
#
# cdef RectRectDistanceTracker dist_tracker
# dist_tracker = RectRectDistanceTracker(rect1, rect2, p)
#
# ...
#
# if dist_tracker.min_distance < ...:
#     ...
#
# dist_tracker.push_less_of(1, node1)
# do_something(node1.less, dist_tracker)
# dist_tracker.pop()
#
# dist_tracker.push_greater_of(1, node1)
# do_something(node1.greater, dist_tracker)
# dist_tracker.pop()

cdef struct RR_stack_item:
    np.intp_t which
    np.intp_t split_dim
    double min_along_dim, max_along_dim
    np.float64_t min_distance, max_distance

cdef np.intp_t LESS = 1
cdef np.intp_t GREATER = 2

cdef class RectRectDistanceTracker(object):
    cdef Rectangle rect1, rect2
    cdef np.float64_t p, epsfac, upper_bound
    cdef np.float64_t min_distance, max_distance

    cdef np.intp_t stack_size, stack_max_size
    cdef RR_stack_item *stack

    # Stack handling
    cdef int _init_stack(self) except -1:
        cdef void *tmp
        self.stack_max_size = 10
        tmp = stdlib.malloc(sizeof(RR_stack_item) *
                            self.stack_max_size)
        if tmp == NULL:
            raise MemoryError
        self.stack = <RR_stack_item*> tmp
        self.stack_size = 0
        return 0

    cdef int _resize_stack(self, np.intp_t new_max_size) except -1:
        cdef void *tmp
        self.stack_max_size = new_max_size
        tmp = stdlib.realloc(<RR_stack_item*> self.stack,
                             new_max_size * sizeof(RR_stack_item))
        if tmp == NULL:
            raise MemoryError
        self.stack = <RR_stack_item*> tmp
        return 0
    
    cdef int _free_stack(self) except -1:
        if self.stack != <RR_stack_item*> NULL:
            stdlib.free(self.stack)
        return 0
    

    def __init__(self, Rectangle rect1, Rectangle rect2,
                 np.float64_t p, np.float64_t eps, np.float64_t upper_bound):
        
        if rect1.m != rect2.m:
            raise ValueError("rect1 and rect2 have different dimensions")

        self.rect1 = rect1
        self.rect2 = rect2
        self.p = p
        
        # internally we represent all distances as distance ** p
        if p != infinity and upper_bound != infinity:
            self.upper_bound = upper_bound ** p
        else:
            self.upper_bound = upper_bound

        # fiddle approximation factor
        if eps == 0:
            self.epsfac = 1
        elif p == infinity:
            self.epsfac = 1 / (1 + eps)
        else:
            self.epsfac = 1 / (1 + eps) ** p

        self._init_stack()

        # Compute initial min and max distances
        if self.p == infinity:
            self.min_distance = min_dist_rect_rect_p_inf(rect1, rect2)
            self.max_distance = max_dist_rect_rect_p_inf(rect1, rect2)
        else:
            self.min_distance = 0.
            self.max_distance = 0.
            for i in range(rect1.m):
                self.min_distance += min_dist_interval_interval_p(rect1, rect2, i, p)
                self.max_distance += max_dist_interval_interval_p(rect1, rect2, i, p)

    def __dealloc__(self):
        self._free_stack()

    cdef int push(self, np.intp_t which, np.intp_t direction,
                  np.intp_t split_dim,
                  np.float64_t split_val) except -1:

        cdef Rectangle rect
        if which == 1:
            rect = self.rect1
        else:
            rect = self.rect2

        # Push onto stack
        if self.stack_size == self.stack_max_size:
            self._resize_stack(self.stack_max_size * 2)
            
        cdef RR_stack_item *item = &self.stack[self.stack_size]
        self.stack_size += 1
        item.which = which
        item.split_dim = split_dim
        item.min_distance = self.min_distance
        item.max_distance = self.max_distance
        item.min_along_dim = rect.mins[split_dim]
        item.max_along_dim = rect.maxes[split_dim]

        # Update min/max distances
        if self.p != infinity:
            self.min_distance -= min_dist_interval_interval_p(self.rect1, self.rect2, split_dim, self.p)
            self.max_distance -= max_dist_interval_interval_p(self.rect1, self.rect2, split_dim, self.p)

        if direction == LESS:
            rect.maxes[split_dim] = split_val
        else:
            rect.mins[split_dim] = split_val

        if self.p != infinity:
            self.min_distance += min_dist_interval_interval_p(self.rect1, self.rect2, split_dim, self.p)
            self.max_distance += max_dist_interval_interval_p(self.rect1, self.rect2, split_dim, self.p)
        else:
            self.min_distance = min_dist_rect_rect_p_inf(self.rect1, self.rect2)
            self.max_distance = max_dist_rect_rect_p_inf(self.rect1, self.rect2)
            
        return 0

    
    cdef inline int push_less_of(self, np.intp_t which,
                                 innernode *node) except -1:
        return self.push(which, LESS, node.split_dim, node.split)

    
    cdef inline int push_greater_of(self, np.intp_t which,
                                    innernode *node) except -1:
        return self.push(which, GREATER, node.split_dim, node.split)

    
    cdef inline int pop(self) except -1:
        # Pop from stack
        self.stack_size -= 1
        assert self.stack_size >= 0
        
        cdef RR_stack_item* item = &self.stack[self.stack_size]
        self.min_distance = item.min_distance
        self.max_distance = item.max_distance

        if item.which == 1:
            self.rect1.mins[item.split_dim] = item.min_along_dim
            self.rect1.maxes[item.split_dim] = item.max_along_dim
        else:
            self.rect2.mins[item.split_dim] = item.min_along_dim
            self.rect2.maxes[item.split_dim] = item.max_along_dim
        
        return 0

# Point-to-rectangle distance tracker
# ===================================
#
# The other logical unit that is used in query_ball_point is to keep track
# of the maximum and minimum distances between points in a hyperrectangle
# and another fixed point as the rectangle is successively split.
#
# Example
# -------
# # node encloses points in rect
#
# cdef PointRectDistanceTracker dist_tracker
# dist_tracker = PointRectDistanceTracker(pt, rect, p)
#
# ...
#
# if dist_tracker.min_distance < ...:
#     ...
#
# dist_tracker.push_less_of(node)
# do_something(node.less, dist_tracker)
# dist_tracker.pop()
#
# dist_tracker.push_greater_of(node)
# do_something(node.greater, dist_tracker)
# dist_tracker.pop()

cdef struct RP_stack_item:
    np.intp_t split_dim
    double min_along_dim, max_along_dim
    np.float64_t min_distance, max_distance

cdef class PointRectDistanceTracker(object):
    cdef Rectangle rect
    cdef np.float64_t *pt
    cdef np.float64_t p, epsfac, upper_bound
    cdef np.float64_t min_distance, max_distance

    cdef np.intp_t stack_size, stack_max_size
    cdef RP_stack_item *stack

    # Stack handling
    cdef int _init_stack(self) except -1:
        cdef void *tmp
        self.stack_max_size = 10
        tmp = stdlib.malloc(sizeof(RP_stack_item) *
                            self.stack_max_size)
        if tmp == NULL:
            raise MemoryError
        self.stack = <RP_stack_item*> tmp
        self.stack_size = 0
        return 0

    cdef int _resize_stack(self, np.intp_t new_max_size) except -1:
        cdef void *tmp
        self.stack_max_size = new_max_size
        tmp = stdlib.realloc(<RP_stack_item*> self.stack,
                              new_max_size * sizeof(RP_stack_item))
        if tmp == NULL:
            raise MemoryError
        self.stack = <RP_stack_item*> tmp
        return 0
    
    cdef int _free_stack(self) except -1:
        if self.stack != <RP_stack_item*> NULL:
            stdlib.free(self.stack)
        return 0

    cdef init(self, np.float64_t *pt, Rectangle rect,
              np.float64_t p, np.float64_t eps, np.float64_t upper_bound):

        self.pt = pt
        self.rect = rect
        self.p = p
        
        # internally we represent all distances as distance ** p
        if p != infinity and upper_bound != infinity:
            self.upper_bound = upper_bound ** p
        else:
            self.upper_bound = upper_bound

        # fiddle approximation factor
        if eps == 0:
            self.epsfac = 1
        elif p == infinity:
            self.epsfac = 1 / (1 + eps)
        else:
            self.epsfac = 1 / (1 + eps) ** p

        self._init_stack()

        # Compute initial min and max distances
        if self.p == infinity:
            self.min_distance = min_dist_point_rect_p_inf(pt, rect)
            self.max_distance = max_dist_point_rect_p_inf(pt, rect)
        else:
            self.min_distance = 0.
            self.max_distance = 0.
            for i in range(rect.m):
                self.min_distance += min_dist_point_interval_p(pt, rect, i, p)
                self.max_distance += max_dist_point_interval_p(pt, rect, i, p)

    def __dealloc__(self):
        self._free_stack()

    cdef int push(self, np.intp_t direction,
                  np.intp_t split_dim,
                  np.float64_t split_val) except -1:

        # Push onto stack
        if self.stack_size == self.stack_max_size:
            self._resize_stack(self.stack_max_size * 2)
            
        cdef RP_stack_item *item = &self.stack[self.stack_size]
        self.stack_size += 1
        
        item.split_dim = split_dim
        item.min_distance = self.min_distance
        item.max_distance = self.max_distance
        item.min_along_dim = self.rect.mins[split_dim]
        item.max_along_dim = self.rect.maxes[split_dim]
            
        if self.p != infinity:
            self.min_distance -= min_dist_point_interval_p(self.pt, self.rect, split_dim, self.p)
            self.max_distance -= max_dist_point_interval_p(self.pt, self.rect, split_dim, self.p)

        if direction == LESS:
            self.rect.maxes[split_dim] = split_val
        else:
            self.rect.mins[split_dim] = split_val

        if self.p != infinity:
            self.min_distance += min_dist_point_interval_p(self.pt, self.rect, split_dim, self.p)
            self.max_distance += max_dist_point_interval_p(self.pt, self.rect, split_dim, self.p)
        else:
            self.min_distance = min_dist_point_rect_p_inf(self.pt, self.rect)
            self.max_distance = max_dist_point_rect_p_inf(self.pt, self.rect)
            
        return 0

    
    cdef inline int push_less_of(self, innernode* node) except -1:
        return self.push(LESS, node.split_dim, node.split)

    
    cdef inline int push_greater_of(self, innernode* node) except -1:
        return self.push(GREATER, node.split_dim, node.split)

    
    cdef inline int pop(self) except -1:
        self.stack_size -= 1
        assert self.stack_size >= 0
        
        cdef RP_stack_item* item = &self.stack[self.stack_size]
        self.min_distance = item.min_distance
        self.max_distance = item.max_distance
        self.rect.mins[item.split_dim] = item.min_along_dim
        self.rect.maxes[item.split_dim] = item.max_along_dim
        
        return 0
        
# Tree structure
# ==============
cdef struct innernode:
    np.intp_t split_dim
    np.intp_t children
    np.float64_t split
    innernode* less
    innernode* greater
    
cdef struct leafnode:
    np.intp_t split_dim
    np.intp_t children
    np.intp_t start_idx
    np.intp_t end_idx


# this is the standard trick for variable-size arrays:
# malloc sizeof(nodeinfo)+self.m*sizeof(np.float64_t) bytes.

cdef struct nodeinfo:
    innernode* node
    np.float64_t side_distances[0]  # FIXME: Only valid in C99, invalid C++ and C89


# Main class
# ==========
cdef class cKDTree2(cKDTree):
    def __init__(self, data, weights=1, leafsize=10):
        super(cKDTree2,self).__init__(data, leafsize)
        if np.shape(weights) == ():
            self.weights = np.zeros(len(self.data))+float(weights)
        elif np.shape(weights) == (len(data),):
            self.weights = np.array(weights,copy=True)
        else: raise ValueError("weights must be a float/int or np.array of floats/ints of len(data)")
        
    # ---------------
    # query_ball_tree_counts
    # ---------------
    cdef int __query_ball_tree_counts_traverse_no_checking(cKDTree2 self,
                                                           cKDTree2 other,
                                                           np.intp_t * results,
                                                           innernode* node1,
                                                           innernode* node2) except -1:
        cdef leafnode *lnode1
        cdef leafnode *lnode2
        cdef list results_i
        cdef np.intp_t i, j
        
        if node1.split_dim == -1:  # leaf node
            lnode1 = <leafnode*>node1
            
            if node2.split_dim == -1:  # leaf node
                lnode2 = <leafnode*>node2
                
                for i in range(lnode1.start_idx, lnode1.end_idx):
                    #results_i = results[self.raw_indices[i]]
                    for j in range(lnode2.start_idx, lnode2.end_idx):
                        #list_append(results_i, other.raw_indices[j])
                        results[self.raw_indices[i]] += 1
                        
            else:
                
                self.__query_ball_tree_counts_traverse_no_checking(other, results, node1, node2.less)
                self.__query_ball_tree_counts_traverse_no_checking(other, results, node1, node2.greater)
        else:
            
            self.__query_ball_tree_counts_traverse_no_checking(other, results, node1.less, node2)
            self.__query_ball_tree_counts_traverse_no_checking(other, results, node1.greater, node2)

        return 0


    @cython.cdivision(True)
    cdef int __query_ball_tree_counts_traverse_checking(cKDTree self, cKDTree other,
                                                        np.intp_t * results,
                                                        innernode* node1,innernode* node2,
                                                        RectRectDistanceTracker tracker) except -1:
        cdef leafnode *lnode1
        cdef leafnode *lnode2
        cdef list results_i
        cdef np.float64_t d
        cdef np.intp_t i, j

        if tracker.min_distance > tracker.upper_bound * tracker.epsfac:
            return 0
        elif tracker.max_distance < tracker.upper_bound / tracker.epsfac:
            self.__query_ball_tree_counts_traverse_no_checking(other, results, node1, node2)
        elif node1.split_dim == -1:  # 1 is leaf node
            lnode1 = <leafnode*>node1
            
            if node2.split_dim == -1:  # 1 & 2 are leaves
                lnode2 = <leafnode*>node2
                
                # brute-force
                for i in range(lnode1.start_idx, lnode1.end_idx):
                    #results_i = results[self.raw_indices[i]]
                    for j in range(lnode2.start_idx, lnode2.end_idx):
                        d = _distance_p(
                            self.raw_data + self.raw_indices[i] * self.m,
                            other.raw_data + other.raw_indices[j] * other.m,
                            tracker.p, self.m, tracker.upper_bound)
                        if d <= tracker.upper_bound:
                            #list_append(results_i, other.raw_indices[j])
                            results[self.raw_indices[i]] += 1
                            
                            
            else:  # 1 is a leaf node, 2 is inner node

                tracker.push_less_of(2, node2)
                self.__query_ball_tree_counts_traverse_checking(
                    other, results, node1, node2.less, tracker)
                tracker.pop()
                    
                tracker.push_greater_of(2, node2)
                self.__query_ball_tree_counts_traverse_checking(
                    other, results, node1, node2.greater, tracker)
                tracker.pop()
            
                
        else:  # 1 is an inner node
            if node2.split_dim == -1:  # 1 is an inner node, 2 is a leaf node
                tracker.push_less_of(1, node1)
                self.__query_ball_tree_counts_traverse_checking(
                    other, results, node1.less, node2, tracker)
                tracker.pop()
                    
                tracker.push_greater_of(1, node1)
                self.__query_ball_tree_counts_traverse_checking(
                    other, results, node1.greater, node2, tracker)
                tracker.pop()
                
            else: # 1 & 2 are inner nodes
                
                tracker.push_less_of(1, node1)
                tracker.push_less_of(2, node2)
                self.__query_ball_tree_counts_traverse_checking(
                    other, results, node1.less, node2.less, tracker)
                tracker.pop()
                    
                tracker.push_greater_of(2, node2)
                self.__query_ball_tree_counts_traverse_checking(
                    other, results, node1.less, node2.greater, tracker)
                tracker.pop()
                tracker.pop()

                
                tracker.push_greater_of(1, node1)
                tracker.push_less_of(2, node2)
                self.__query_ball_tree_counts_traverse_checking(
                    other, results, node1.greater, node2.less, tracker)
                tracker.pop()
                    
                tracker.push_greater_of(2, node2)
                self.__query_ball_tree_counts_traverse_checking(
                    other, results, node1.greater, node2.greater, tracker)
                tracker.pop()
                tracker.pop()
            
        return 0
            

    def query_ball_tree_counts(cKDTree self, cKDTree other, np.float64_t r,
                               np.float64_t p=2., np.float64_t eps=0):
        """query_ball_tree(self, other, r, p, eps)

        Find all pairs of points whose distance is at most r

        Parameters
        ----------
        other : KDTree instance
            The tree containing points to search against.
        r : float
            The maximum distance, has to be positive.
        p : float, optional
            Which Minkowski norm to use.  `p` has to meet the condition
            ``1 <= p <= infinity``.
        eps : float, optional
            Approximate search.  Branches of the tree are not explored
            if their nearest points are further than ``r/(1+eps)``, and
            branches are added in bulk if their furthest points are nearer
            than ``r * (1+eps)``.  `eps` has to be non-negative.

        Returns
        -------
        results : list of lists
            For each element ``self.data[i]`` of this tree, ``results[i]`` is a
            list of the indices of its neighbors in ``other.data``.

        """

        # Make sure trees are compatible
        if self.m != other.m:
            raise ValueError("Trees passed to query_ball_tree have different dimensionality")

        # Track node-to-node min/max distances
        tracker = RectRectDistanceTracker(
            Rectangle(self.mins, self.maxes),
            Rectangle(other.mins, other.maxes),
            p, eps, r)
        
        cdef np.ndarray[np.intp_t, ndim=1, mode="c"] results
        #results = [[] for i in range(self.n)]
        #results = [0 for i in range(self.n)]
        results = np.zeros((self.n,), dtype=np.intp)
        self.__query_ball_tree_counts_traverse_checking(other, &results[0], self.tree,
                                                        other.tree, tracker)

        return results
        
        