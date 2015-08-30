from numpy import *
import theano
import theano.tensor as T
import theano.typed_list as tl
import theano.sparse as sparse
from scipy.misc import logsumexp
from scipy.optimize import fmin_ncg
import scipy.sparse as sp
import time


random.seed(1)

K = 5 #nClasses
N = 150 #nSamples
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
tP = T.nnet.softmax(tEta)
terror = T.nnet.categorical_crossentropy(tP, ty).mean() + tlambda * tW.norm(2)**2 # we could add some Tikhonov regularization
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
E = eye(K-1).astype(float32)
idx2 = [ sp.csc_matrix(E[it,:]) for it in idx]
labels = []
labels.append(array([-1,-1,-1], dtype=float32))
labels.append(array([-1,-1,1], dtype=float32))
labels.append(array([-1,1,-1], dtype=float32))
labels.append(array([-1,1,1], dtype=float32))
labels.append(array([1], dtype=float32))

#hierarchical parameters
Wh = random.randn(K-1,D).astype(float32)
tWh = T.matrix()
tlIdx = tl.TypedListType(T.lvector)()
tlIdx2 = tl.TypedListType(sparse.csc_fmatrix)()
tlLab = tl.TypedListType(T.fvector)()

test = tlIdx[0]
print test.eval({tlIdx:idx})

tr = T.lvector()

#we can generate outputs as well as shared var updates by returning dictionary
n = 0
tsh = theano.shared(n)
gWh = zeros(Wh.shape).astype(float32)
tgWh = theano.shared(gWh)

def run(i,xt, L, W, labels, tsh):
    ft= -T.log1p(T.exp(-labels[i] * T.dot(W[L[i],:], xt))).sum()
    bla = T.grad(ft, W)
    return ft,  {tsh:tsh+1}
   

def run2(i,xt, L, W, labels, tsh):
    Wtemp = sparse.structured_dot(L[i], W)
    ft = T.log1p(T.exp(-labels[i] * T.dot(Wtemp, xt))).sum()
    gt = T.grad(ft, W)
    #return ft, {tgWh:tgWh+sparse.true_dot(L[i].T, gt)}
    return ft, {tgWh:tgWh+W}

s,u = theano.scan(run, sequences=[tr, tX], non_sequences=[tlIdx, tWh, tlLab, tsh])
s2,u2 = theano.scan(run2, sequences=[tr, tX], non_sequences=[tlIdx2, tWh, tlLab, tgWh])

obj2 = s2.sum()

bla = theano.function([tr,tX, tlIdx, tWh, tlLab], s, updates=u)
bla2 = theano.function([tr,tX, tlIdx2, tWh, tlLab], s2, updates=u2)
bla3 = theano.function([tr,tX, tlIdx2, tWh, tlLab], obj2, updates=u2)
xx = random.randn(5, D).astype(float32)
hey4 = bla(arange(5),xx, idx, Wh, labels)
hey3 = bla2(arange(5),xx, idx2, Wh, labels)
hey2 = s.mean().eval({tr:y, tX:X, tlIdx:idx, tWh:Wh, tlLab:labels})
tic = time.time()
hey = bla(y, X, idx, Wh, labels)
print time.time() - tic

tic = time.time()
res= []
for it in y:#xrange(5):
    temp = labels[it] * dot(Wh[ idx[it], :], xx[it,:])
    res.append( -log1p(exp(-temp)).sum() )
print time.time() - tic

zz = zeros(Wh.shape).astype(float32)

for it in xrange(50):
    tgWh.set_value(zz) #this will hold the gradient
    ft = bla3(y, X, idx2, Wh, labels)
    Wh -= 0.1 * tgWh.get_value() / X.shape[0]
    print ft/X.shape[0], linalg.norm(tgWh.get_value(), ord='fro')

#tsize = T.matrix()
#tV = T.matrix()
#tI = T.ivector()
#bla = sparse.construct_sparse_from_list(tsize, tV, tI)
#
#
#v = ones(2).astype(float32)
#i = arange(2)
#j = arange(2)
#m = sp.csc_matrix( (v, (i,j)), shape=(4,2))
#
#
#tdata = T.vector()
#tindices = T.ivector()
#tindptr = T.ivector()
#tshape = T.ivector()
#x = sparse.csc_matrix()
#a,b,c,d = sparse.csm_properties(x)
#print a.eval({x:m})
#print b.eval({x:m})
#print c.eval({x:m})
#print d.eval({x:m})
#
#tm = sparse.CSC(tdata, tindices, tindptr, tshape)
#
#shape = array([5,3]).astype(int32)
#indices = array( [0,2,4] ).astype(int32)
#indptr = arange(4).astype(int32)
#data = ones(3).astype(float32)
#m2 = tm.eval( {tdata:data, tindices:indices, tindptr:indptr, tshape:shape})
#
#ty = T.ivector()
#
#def hsm(tyt, txt, tidx, tWh):
#    temp = T.dot(tWh[tidx,:], txt)
#    
#
#
##s,u = theano.scan(lambda tyt,txt, tlIdx, tWh: hsm(tyt,txt, tlIdx[tyt], tWh), sequences=[ty, tX], non_sequences=[tlIdx, tWh])
#
#
#
#
