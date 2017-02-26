import theano
import numpy
import cPickle
import theano.tensor as T
from theano.tensor.shared_randomstreams import RandomStreams

class HiddenLayer:
    def __init__(self,
                 rng,                   # Random seed
                 input,                 # Data input
                 numIn,                 # Number neurons of input
                 numOut,                # Number reurons out of layer
                 activation = T.tanh,   # Activation function
                 W = None,
                 b = None,
                 corruption = None
                 ):
        # Set parameters
        self.Rng = rng;
        self.Input = input
        self.NumIn = numIn
        self.NumOut = numOut
        self.Activation = activation
        self.Corruption = corruption

        # Create shared parameters for hidden layer
        if W is None:
            """ We create random weights (uniform distribution) """
            # Create boundary for uniform generation
            wBound = numpy.sqrt(6.0 / (self.NumIn + self.NumOut))
            self.W = theano.shared(
                numpy.asarray(
                    rng.uniform(
                        low=-wBound,
                        high=wBound,
                        size=(self.NumIn, self.NumOut)
                    ),
                    dtype=theano.config.floatX
                ),
                borrow=True
            )
        else:
            """ Or simply set weights from parameter """
            self.W = W

        if b is None:
            """ We create zeros bias """
            # Create bias
            self.b = theano.shared(
                numpy.zeros(
                    shape = (self.NumOut, ),
                    dtype=theano.config.floatX
                ),
                borrow=True
            )
        else:
            """ Or simply set bias from parameter """
            self.b = b

    def getCorruptedInput(self, input, corruptionLevel):
        theano_rng = RandomStreams(self.Rng.randint(2 ** 30))
        return theano_rng.binomial(size=input.shape, n=1,
                                   p=1 - corruptionLevel,
                                   dtype=theano.config.floatX) * input

    def Output(self):
        input = self.Input
        if self.Corruption is not None:
            self.Input = self.getCorruptedInput(self.Input, self.Corruption)
        output = T.dot(input, self.W) + self.b
        if self.Activation is None:
            return output
        else:
            return self.Activation(output)

    '''
    Return transpose of weight matrix
    '''
    def WTranspose(self):
        return self.W.T

    def Params(self):
        return [self.W, self.b]

    def LoadModel(self, file):
        self.W.set_value(cPickle.load(file), borrow = True)
        self.b.set_value(cPickle.load(file), borrow = True)