from numpy import *
import theano
import theano.tensor as T
import theano.typed_list as tl
import theano.sparse as sparse
from scipy.misc import logsumexp
from scipy.optimize import fmin_ncg
import scipy.sparse as sp
import time

"""
Explore how to implement custom theano ops.
For now Python only
"""

class MySoftmax(theano.gof.Op):

    __props__ = ()

    def make_node(self, Eta):
        return theano.Apply(self, [Eta], [Eta.type(), Eta.type()])

    def perform(self, node, input_storage, output_storage):
        print "perform"
        Eta = input_storage[0]
        lNorm = logsumexp(Eta, axis=1).reshape(N,1)
        lP = Eta - lNorm 
        output_storage[0][0] = exp(lP)
        output_storage[1][0] = exp(lP)

    def grad(self, inputs, g):
        print "gradnick"
        Eta = input_storage[0]

mysoftmax = MySoftmax()

random.seed(1)

K = 5 #nClasses
N = 6 #nSamples
D = 3 #nFeatures

#single precision for now
theano.config.floatX = 'float32'
theano.config.scan.allow_gc = False

#setup toy example
W = random.randn(D,K)
X = random.randn(N,D)
Eta = dot(X,W)
lNorm = logsumexp(Eta, axis=1).reshape(N,1)
lP = Eta - lNorm 
#take one sample from a multinomial distribution specified by a row of lP
_,y = apply_along_axis(lambda row: random.multinomial(1, exp(row)), axis=1, arr=lP).nonzero()
W = W.astype(float32)
X = X.astype(float32)
y = y.astype(int32)

#setup theano
tW = T.matrix('W')
tX = T.matrix('X')
ty = T.ivector('y')
tlambda = T.scalar('lambda')

#symbolic representation
tEta = T.dot(tX, tW)
Eta = Eta.astype(float32)
b = theano.function([tEta], mysoftmax(tEta))
c = b(Eta)
tP = T.nnet.softmax(tEta)
tP2 = mysoftmax(tEta)
hey = theano.function( [tX, tW], tP2)
terror = T.nnet.categorical_crossentropy(tP, ty).mean() #+ tlambda * tW.norm(2)**2 # we could add some Tikhonov regularization
terr2 = T.nnet.categorical_crossentropy(tP2[0], ty).mean() 
tgrad = T.grad(terror, tW)
#f = theano.function([tW, tX, ty, tlambda], terror)
#g = theano.function([tW, tX, ty, tlambda], tgrad)
