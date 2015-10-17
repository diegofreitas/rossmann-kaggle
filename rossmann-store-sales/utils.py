__author__ = 'diego.freitas'

import numpy as np
def to_weight(y):
    w = np.zeros(y.shape, dtype=float)
    ind = y != 0
    w[ind] = 1./(y[ind]**2)
    return w

def rmspe( y,yhat):
    w = to_weight(y)
    rmspe = np.sqrt(np.mean( w * (y - yhat)**2 ))
    return rmspe

def rmspe_xg(y, yhat):
    y = np.exp(y) - 1
    yhat = np.exp(yhat) - 1
    w = to_weight(y)
    rmspe = np.sqrt(np.mean(w * (y - yhat)**2))
    return "rmspe", rmspe
