from numpy import *
import theano
import theano.tensor as T
import theano.typed_list as tl
import theano.sparse as sparse
from scipy.misc import logsumexp
from scipy.optimize import fmin_ncg
import scipy.sparse as sp
import time

class HierarchicalSoftmax(theano.gof.Op):

    __props__ = ()

    def make_node(self, *inputs):
        H, W = inputs
        #we get the hidden state and the output layer weights and want to compute the log probability of the output and the gradient wrt. the output weights W
        return theano.Apply(self, [W,H], [W.type(), W.type()])

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


if __name__ == "__main__":
#do some testing here
    print "hsm.py"
    hsm = HierarchicalSoftmax
