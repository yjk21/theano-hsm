from numpy import *
import theano
import theano.tensor as T
from scipy.misc import logsumexp
from scipy.optimize import fmin_ncg

random.seed(1)

K = 5#nClasses
N = 10 #nSamples
Dout = 4 #hidden features
Din = 2 #input features

#single precision for now
theano.config.floatH = 'float32'


#setup toy example
X = random.randn(N, Din)
Win = random.randn(Din, Dout)
Wout = random.randn(Dout,K)
#hidden activations
Hlin = dot(X, Win)
H = tanh(Hlin)
Eta = dot(H,Wout)
lNorm = logsumexp(Eta, axis=1).reshape(N,1)
lP = Eta - lNorm 
#take one sample from a multinomial distribution specified by a row of lP
_,y = apply_along_axis(lambda row: random.multinomial(1, exp(row)), axis=1, arr=lP).nonzero()
y = y.astype(int32)
Wout = Wout.astype(float32)
Win = Win.astype(float32)
H = H.astype(float32)
X = X.astype(float32)
Hlin = Hlin.astype(float32)

#setup theano
tX = T.matrix('X')
tWin = T.matrix('Win')
tWout = T.matrix('Wout')
ty = T.ivector('y')
tlambda = T.scalar('lambda')

#symbolic representation
tHlin = T.dot(tX, tWin)
tH = T.tanh(tHlin)
tEta = T.dot(tH, tWout)
tP = T.nnet.softmax(tEta)
terror = T.nnet.categorical_crossentropy(tP, ty).mean()  
tgrad = T.grad(terror, [tWout, tWin])

#numeric functions
f = theano.function([tWout, tWin, tX, ty], terror)
g = theano.function([tWout, tWin, tX, ty], tgrad)

"""
Main Part of experiment:
    Can we manually invoke the chain rule to let theano compute gradients deeper down in the architecture?
    The goal is to compute the gradient wrt. parameters of the output layer by hand which is more convenient and then let theano figure out the rest, i.e. the parameters of the RNN backbone automatically
"""
#Compute reference gradient
gWout, gWin = g(Wout, Win, X,y)

#compute symbolic graident wrt. to hidden
tgH = T.grad(terror, tH)

#invoking chain rule manually with partial symbolic gradient with respect to hidden unit
tgWin2 = theano.gradient.Lop(tH, tWin, tgH)
gWin2 = tgWin2.eval({tX:X,ty:y,tWin:Win,tWout:Wout})

#can we do it also with an actual partial numeric result?
gH = tgH.eval({tX:X,ty:y,tWin:Win,tWout:Wout})
tgHdummy = T.matrix("tgHdummy") 
tgWin3 = theano.gradient.Lop(tH, tWin, tgHdummy)
gWin3 = tgWin3.eval({tX:X,tWin:Win,test:gH})
