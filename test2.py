from numpy import *
import theano
import theano.tensor as T
import theano.typed_list as tl
import theano.sparse as sparse
from scipy.misc import logsumexp
from scipy.optimize import fmin_ncg
import scipy.sparse as sp


random.seed(1)

K = 5 #nClasses
N = 50 #nSamples
D = 4 #nFeatures

#single precision for now
theano.config.floatX = 'float32'

#setup toy example
W = random.randn(D,K).astype(float32)
X = random.randn(N,D).astype(float32)
Eta = dot(X,W)
lNorm = logsumexp(Eta, axis=1).reshape(N,1)
lP = Eta - lNorm 
#take one sample from a multinomial distribution specified by a row of lP
_,y = apply_along_axis(lambda row: random.multinomial(1, exp(row)), axis=1, arr=lP).nonzero()
y = y.astype(int32)

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
#f = theano.function([tW, tX, ty, tlambda], terror)
#g = theano.function([tW, tX, ty, tlambda], tgrad)
#
#W0 = random.randn(D,K).astype(float32) 
#
##gradient descent
#for it in xrange(5):
#    ft = f(W0, X, y, 0.1)
#    gt = g(W0, X, y, 0.1)
#    W0 -= 0.1 * gt 
#    print it, "objective:",ft, "gradnorm:",linalg.norm(gt, ord=inf)


#define mapping by hand
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

#hierarchical parameters
Wh = random.randn(D,K-1).astype(float32)
tWh = T.matrix()
tlIdx = tl.TypedListType(T.lvector)()
test = tlIdx[0]
print test.eval({tlIdx:idx})
tr = T.lvector()
def act(idx, W):
    M = W.shape[0]
    N = W.shape[1]
    K = idx.shape[0]
    data = T.ones(K)
    I = sparse.csc_matrix()
    return sparse.true_dot(I, W[idx,:])
s,u = theano.scan(lambda i, L, D: D[ L[i],: ], sequences=[tr], non_sequences=[tlIdx, tWh])

bla = theano.function([tr, tlIdx, tWh], s)
hey = bla(arange(5), idx, Wh)


tsize = T.matrix()
tV = T.matrix()
tI = T.ivector()
bla = sparse.construct_sparse_from_list(tsize, tV, tI)


v = ones(2).astype(float32)
i = arange(2)
j = arange(2)
m = sp.csc_matrix( (v, (i,j)), shape=(4,2))


tdata = T.vector()
tindices = T.ivector()
tindptr = T.ivector()
tshape = T.ivector()
x = sparse.csc_matrix()
a,b,c,d = sparse.csm_properties(x)
print a.eval({x:m})
print b.eval({x:m})
print c.eval({x:m})
print d.eval({x:m})

tm = sparse.CSC(tdata, tindices, tindptr, tshape)

shape = array([5,3]).astype(int32)
indices = array( [0,2,4] ).astype(int32)
indptr = arange(4).astype(int32)
data = ones(3).astype(float32)
m2 = tm.eval( {tdata:data, tindices:indices, tindptr:indptr, tshape:shape})

ty = T.ivector()

def hsm(tyt, txt, tidx, tWh):
    temp = T.dot(tWh[tidx,:], txt)
    


s,u = theano.scan(lambda tyt,txt, tlIdx, tWh: hsm(tyt,txt, tlIdx[tyt], tWh), sequences=[ty, tX], non_sequences=[tlIdx, tWh])




