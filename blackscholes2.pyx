"""Fast Cython based Black-Scholes options pricing.

To use this interactively, just do::

    >>> import pyximport
    >>> pyximport.install()
    >>> from blackscholes2 import black_scholes
    >>> black_scholes(100.0, 100.0, 1.0, 0.3, 0.03, 0.0, -1)
    10.327861752731728
"""

cdef extern from "math.h" nogil:
    double exp(double)
    double sqrt(double)
    double pow(double, double)
    double log(double)
    double erf(double)

cdef double std_norm_cdf(double x):
    return 0.5*(1+erf(x/sqrt(2.0)))

def black_scholes(double s, double k, double t, double v,
                 double rf, double div, double cp):
    """Price an option using the Black-Scholes model.
    
    s : initial stock price
    k : strike price
    t : expiration time
    v : volatility
    rf : risk-free rate
    div : dividend
    cp : +1/-1 for call/put
    """
    cdef double d1, d2, optprice
    d1 = (log(s/k)+(rf-div+0.5*pow(v,2))*t)/(v*sqrt(t))
    d2 = d1 - v*sqrt(t)
    optprice = cp*s*exp(-div*t)*std_norm_cdf(cp*d1) - \
        cp*k*exp(-rf*t)*std_norm_cdf(cp*d2)
    return optprice

