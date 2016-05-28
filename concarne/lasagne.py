# -*- coding: utf-8 -*-

from __future__ import absolute_import

#__all__ = ["MultivariateDenseLayer", "multivariate_categorical_cross_entropy"]

import lasagne
from lasagne import nonlinearities
from lasagne import init

import theano
import theano.tensor as T

import numpy as np

class MultivariateDenseLayer(lasagne.layers.Layer):
    """
    concarne.lasagne.MultivariateDenseLayer(incoming, num_units,
    W0=lasagne.init.GlorotUniform(), b0=lasagne.init.Constant(0.),
    W1=lasagne.init.GlorotUniform(), b1=lasagne.init.Constant(0.),
    nonlinearity=lasagne.nonlinearities.softmax, **kwargs)

    A set of fully connected layers.
    
    It is most useful in combination with the multivariate_categorical_crossentropy
    objective.

    Parameters
    ----------
    incoming : a :class:`Layer` instance or a tuple
        The layer feeding into this layer, or the expected input shape

    num_units_per_var : list of int
        The number of units for each layer

    Wi : Theano shared variable, numpy array or callable (i=0...)
        An initializer for weights of the i-th. If a shared variable or a
        numpy array is provided the shape should  be (num_inputs, num_units).
        See :meth:`Layer.create_param` for more information.

    bi : Theano shared variable, numpy array, callable or None (i=0...)
        An initializer for the biases of the i-th layer. If a shared variable or a
        numpy array is provided the shape should be (num_units,).
        If None is provided the layer will have no biases.
        See :meth:`Layer.create_param` for more information.

    nonlinearity : callable or None
        The nonlinearity that is applied to the layer activations. If None
        is provided, the layer will use softmax.

    Examples
    --------
    >>> from lasagne.layers import InputLayer, DenseLayer
    >>> from concarne import MultivariateDenseLayer
    >>> l_in = InputLayer((100, 20))
    >>> l1 = concarne.lasagne.MultivariateDenseLayer(l_in, num_units_per_var=[50, 75])

    Notes
    -----
    If the input to this layer has more than two axes, it will flatten the
    trailing axes. This is useful for when a dense layer follows a
    convolutional layer, for example. It is not necessary to insert a
    :class:`FlattenLayer` in this case.
    """
    def __init__(self, incoming, num_units_per_var, 
                 nonlinearity=nonlinearities.softmax,
                 **kwargs):
        super(MultivariateDenseLayer, self).__init__(incoming, **kwargs)
        
        self.nonlinearity = (nonlinearities.identity if nonlinearity is None
                             else nonlinearity)

        self.num_units_per_var = num_units_per_var
        self.num_vars = len(num_units_per_var)

        num_inputs = int(np.prod(self.input_shape[1:]))
        
        # generate Wi and bi
        for i in range(self.num_vars):
            # W
            mem_str = "W%d" % (i)
            if mem_str not in kwargs:
                # default values
                kwargs[mem_str] = init.GlorotUniform()
                                
                self.__dict__[mem_str] = \
                    self.add_param(kwargs[mem_str], (num_inputs, num_units_per_var[i]), name=mem_str)

            # b
            mem_str = "b%d" % (i)
            if mem_str not in kwargs:
                # default values
                kwargs[mem_str] = init.Constant(0.)

                self.__dict__[mem_str] = \
                    self.add_param(kwargs[mem_str], (num_units_per_var[i],), name=mem_str, regularizable=False)

    def __get_var_member(self, member, num):
        params = vars(self)
        mem_str = "%s%d" % (member, num)
        if mem_str not in params:
            raise Exception("%s not a member" % mem_str)
        return params[mem_str]

    def get_output_shape_for(self, input_shape):
        output_shapes = []
        for num_units in self.num_units_per_var:
            output_shapes.append( (input_shape[0], num_units) )
        return output_shapes

    def get_output_for(self, input, **kwargs):
        if input.ndim > 2:
            # if the input has more than two dimensions, flatten it into a
            # batch of feature vectors.
            input = input.flatten(2)

        outputs = []
        
        for i in range(self.num_vars):
            W = self.__get_var_member("W", i)
            b = self.__get_var_member("b", i)
            
            activation = T.dot(input, W)
            if b is not None:
                activation = activation + b.dimshuffle('x', 0)
            outputs.append(self.nonlinearity(activation))
        
        #return T.concatenate(outputs, axis=1)
        return outputs

# ------------------------------

def multivariate_categorical_crossentropy(predictions, targets):
    """Extension of categorical_crossentropy for a multivariate target.
    
    Computes the categorical cross-entropy between a list of predictions and
        a multivariate target.

    .. math:: L_i = - \\sum_k \\sum_j{t_{i,j}^k \\log(p_{i,j}^k)}
    
    Parameters
    ----------
    predictions : list of Theano 2D tensors
        Predictions in (0, 1), such as softmax output of a neural network,
        with data points in rows and class probabilities in columns.
    targets : Theano 2D tensor
        Either targets in [0, 1] matching the layout of `predictions`, or
        a matrix of int giving the correct class index per data point per target.

    Returns
    -------
    Theano 1D tensor
        An expression for the target-wise and item-wise categorical cross-entropy.

    Notes
    -----
    This is the loss function of choice for multi-class classification
    problems and softmax output units. For hard targets, i.e., targets
    that assign all of the probability to a single class per data point,
    providing a vector of int for the targets is usually slightly more
    efficient than providing a matrix with a single 1.0 per row.
    """
    #num_results = len(predictions)
    
    losses = 0.
    for i, pred in enumerate(predictions):
        losses += theano.tensor.nnet.categorical_crossentropy(pred, targets[:,i])
    return losses

def argmax_multivariate_categorical_crossentropy(predictions):
    return [T.argmax(p, axis=1) for (i, p) in enumerate(predictions)]

def score_categorical_crossentropy(prediction, target):
    return T.mean(T.eq(T.argmax(prediction, axis=1), target), dtype=theano.config.floatX)    

def score_multivariate_categorical_crossentropy(predictions, target):
    """
        import theano
        import theano.tensor as T
        inputs = T.matrix("inputs")
        targets = T.imatrix("targets")
        network = ...
        prediction = lasagne.layers.get_output(network, deterministic=True)
        
        score_fn = theano.function([inputs, targets], score_multivariate_categorical_crossentropy(prediction, targets))
    """
    return T.mean(T.concatenate([ T.eq(T.argmax(p, axis=1), target[:,i])
           for (i, p) in enumerate(predictions)]), dtype=theano.config.floatX)

# ------------------------------
