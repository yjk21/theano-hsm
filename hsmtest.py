from numpy import *
import theano
import theano.tensor as T
import theano.typed_list as tl
import theano.sparse as sparse
from scipy.misc import logsumexp
from scipy.optimize import fmin_ncg
import scipy.sparse as sp

random.seed(1)

idx = []
idx.append(array([0,1,2],dtype=int64))
idx.append(idx[0])
idx.append(array([0,1,3],dtype=int64))
idx.append(idx[2])
idx.append(array([0],dtype=int64))
labels = []
labels.append(array([-1,-1,-1], dtype=float32))
labels.append(array([-1,-1,1], dtype=float32))
labels.append(array([-1,1,-1], dtype=float32))
labels.append(array([-1,1,1], dtype=float32))
labels.append(array([1], dtype=float32))

def sigma(x):
    return 1/(1+exp(-x))
def dsigma(x):
    s = sigma(x)
    return s * (1-s)

def logsig(x):
    return -log1p(exp(-x))
def dlogsig(x):
    return -(1-sigma(x))

def hsm(W, H, y, idx, labels):
    lp = 0.0
    for it, yt in enumerate(y):
        idxt = idx[yt]
        Wi = W[idxt,:]
        Eta = t * dot(Wi, H[:,it])
        lp += sum( sigma(Eta) )
    return lp


