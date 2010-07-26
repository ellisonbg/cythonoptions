"""Fast Cython version of binomial options pricing that releases the GIL.

This also has array bounds and wrap-around checking disabled.

To use this interactively, just do::

    >>> import pyximport, numpy
    >>> pyximport.install(setup_args=dict(include_dirs=[numpy.get_include()]))
    >>> from binomial4 import jarrow_rudd
    >>> jarrow_rudd(100.0, 100.0, 1.0, 0.3, 0.03, -1, True, 100)
    10.627682380924412
"""
import numpy as np
import math

cimport numpy as np
cimport cython

cdef extern from "math.h" nogil:
    double exp(double)
    double sqrt(double)
    double pow(double, double)
    double fmax(double, double)

@cython.boundscheck(False) 
@cython.wraparound(False)
@cython.cdivision(True)
def jarrow_rudd(double s, double k, double t, double v, 
                double rf, double cp, int am=0, int n=100):
    """Price an option using the Jarrow Rudd binomial model.
    
    s : initial stock price
    k : strike price
    t : expiration time
    v : volatility
    rf : risk-free rate
    cp : +1/-1 for call/put
    am : 1/0 for American/European
    n : binomial steps
    """
    cdef double h, u, d, drift, q
    cdef int i, j, m
    cdef np.ndarray[np.double_t, ndim=2] stkval = np.zeros((n+1,n+1))
    cdef np.ndarray[np.double_t, ndim=2] optval = np.zeros((n+1,n+1))

    with nogil:
        #Basic calculations
        h = t/n
        u = exp((rf-0.5*pow(v,2))*h+v*sqrt(h))
        d = exp((rf-0.5*pow(v,2))*h-v*sqrt(h))
        drift = exp(rf*h)
        q = (drift-d)/(u-d)

        #Process the terminal stock price
        stkval[0,0] = s
        for i in range(1,n+1):
            stkval[i,0] = stkval[i-1,0]*u
            for j in range(1,i+1):
                stkval[i,j] = stkval[i-1,j-1]*d

        #Backward recursion for option price
        for j in range(n+1):
            optval[n,j] = fmax(0,cp*(stkval[n,j]-k))
        for m in range(n):
            i = n-m-1
            for j in range(i+1):
                optval[i,j] = (q*optval[i+1,j]+(1-q)*optval[i+1,j+1])/drift
                if am==1:
                    optval[i,j] = fmax(optval[i,j],cp*(stkval[i,j]-k))

    return optval[0,0]

