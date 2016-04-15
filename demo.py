from numpy import *
import theano
import theano.tensor as T
from scipy.misc import logsumexp
from scipy.optimize import fmin_ncg

random.seed(1)

K = 5 #nClasses
N = 100 #nSamples
D = 4 #nFeatures

#single precision for now
theano.config.floatX = 'float32'

#setup toy example
W = random.randn(D,K)
X = random.randn(N,D)
Eta = dot(X,W)
lNorm = logsumexp(Eta, axis=1).reshape(N,1)
lP = Eta - lNorm 
#take one sample from a multinomial distribution specified by a row of lP
_,y = apply_along_axis(lambda row: random.multinomial(1, exp(row)), axis=1, arr=lP).nonzero()
y = y.astype(int32)
W = W.astype(float32)
X = X.astype(float32)

#setup theano
tW = T.matrix('W')
tX = T.matrix('X')
ty = T.ivector('y')
tlambda = T.scalar('lambda')

#symbolic representation
tEta = T.dot(tX, tW)
tP = T.nnet.softmax(tEta)
terror = T.nnet.categorical_crossentropy(tP, ty).mean()  + tlambda * tW.norm(2)**2 # we could add some Tikhonov regularization
tgrad = T.grad(terror, tW)
f = theano.function([tW, tX, ty, tlambda], terror)
g = theano.function([tW, tX, ty, tlambda], tgrad)

W0 = random.randn(D,K).astype(float32) 

#gradient descent
for it in xrange(500):
    ft = f(W0, X, y, 0.1)
    gt = g(W0, X, y, 0.1)
    W0 -= 0.1 * gt 
    print it, "objective:",ft, "gradnorm:",linalg.norm(gt, ord=inf)

