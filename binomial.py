"""Functions to price calls and puts with a binomial tree."""
import numpy as np
import math

def jarrow_rudd(s, k, t, v, rf, cp, am=False, n=100):
    """Price an option using the Jarrow Rudd binomial model.
    
    s : initial stock price
    k : strike price
    t : expiration time
    v : volatility
    rf : risk-free rate
    cp : +1/-1 for call/put
    am : True/False for American/European
    n : binomial steps
    """
    #Basic calculations
    h = t/n
    u = math.exp((rf-0.5*math.pow(v,2))*h+v*math.sqrt(h))
    d = math.exp((rf-0.5*math.pow(v,2))*h-v*math.sqrt(h))
    drift = math.exp(rf*h)
    q = (drift-d)/(u-d)

    #Process the terminal stock price
    stkval = np.zeros((n+1,n+1))
    optval = np.zeros((n+1,n+1))
    stkval[0,0] = s
    for i in range(1,n+1):
        stkval[i,0] = stkval[i-1,0]*u
        for j in range(1,i+1):
            stkval[i,j] = stkval[i-1,j-1]*d

    #Backward recursion for option price
    for j in range(n+1):
        optval[n,j] = max(0,cp*(stkval[n,j]-k))
    for i in range(n-1,-1,-1):
        for j in range(i+1):
            optval[i,j] = (q*optval[i+1,j]+(1-q)*optval[i+1,j+1])/drift
            if am:
                optval[i,j] = max(optval[i,j],cp*(stkval[i,j]-k))

    return optval[0,0]

