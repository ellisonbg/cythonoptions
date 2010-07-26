"""Functions to price calls and puts with a binomial tree"""
import math
import numpy as np
cimport numpy as np
DTYPE = np.double
ctypedef np.double_t DTYPE_t

"""JARROW RUDD MODEL"""
#Enter cp = +/- 1 for call/put
#Enter am = 1/0 for American/European
# Uncomment the following two lines to turn off bounds checking for speedup
# cimport cython
# @cython.boundscheck(False) # turn of bounds-checking for entire function
def jr(double s, double k, double t, double v, double rf, double cp, int am, int n):
    #Basic calculations
    cdef double h, u, d, drift, q
    cdef int i, j
    
    h = t/n
    u = math.exp((rf-0.5*math.pow(v,2))*h+v*math.sqrt(h))
    d = math.exp((rf-0.5*math.pow(v,2))*h-v*math.sqrt(h))
    drift = math.exp(rf*h)
    q = (drift-d)/(u-d)

    #Process the terminal stock price
    cdef np.ndarray[DTYPE_t, ndim=2] stk = np.zeros((n+1,n+1), dtype=DTYPE)
    cdef np.ndarray[DTYPE_t, ndim=2] optval = np.zeros((n+1,n+1), dtype=DTYPE)
    stk[0,0] = s
    for i in range(1,n+1):
        stk[i,0] = stk[i-1,0]*u
        for j in range(1,i+1):
            stk[i,j] = stk[i-1,j-1]*d
    
    #Backward recursion for option price
    for j in range(n+1):
        optval[n,j] = max(0,cp*(stk[n,j]-k))
    for i in range(n-1,-1,-1):
        for j in range(i+1):
            optval[i,j] = (q*optval[i+1,j]+(1-q)*optval[i+1,j+1])/drift
            if am==1:
                optval[i,j] = max(optval[i,j],cp*(stk[i,j]-k))
         
    return optval[0,0]

    


