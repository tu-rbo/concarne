# -*- coding: utf-8 -*-

#!/usr/bin/env python

from __future__ import print_function

import concarne
import concarne.patterns

import lasagne
import theano
import theano.tensor as T

import numpy as np


def build_linear_simple(input_layer, n_out, nonlinearity=None, name=None):
    network = lasagne.layers.DenseLayer(input_layer, n_out, nonlinearity=nonlinearity, b=None, name=name)
    return network    


class TestBase(object):
    """
      Setup class generating some simplistic direct data
    """
    
    def __init__(self):
        self.loss_weights = {'target_weight':0.9, 'context_weight':0.1}
    
    def setup(self):
        self.X = np.array( [[0,1,2,3,4], [1,1,1,1,1]]).T
        self.S = self.X[:,:1]
        self.Y = np.array( [0, 0, 1, 1, 1], dtype='int32' )
        
        self.n = self.X.shape[1]
        
        self.init_variables()
        self.build_pattern()

    def init_variables(self):
        self.input_var = T.matrix('inputs')
        self.context_var = T.matrix('contexts')
        #self.num_classes = 2
        #self.target_var = T.ivector('targets')
        # do regression
        self.num_classes = 1
        self.target_var = T.vector('targets')

    def build_pattern(self):
        raise NotImplemented()

    def build_target_loss(self):
        phi_output = self.phi.get_output_for(self.input_var)
        psi_output = self.psi.get_output_for(phi_output).reshape((-1,))
        self.target_loss = lasagne.objectives.squared_error(psi_output, self.target_var).mean()


# ------------------------------------------------------------------------------

class TestDirectPattern(TestBase):
    """
      Test the direct pattern with linear models
    """

    def setup(self):
        self.C = np.array( [[0,1,2,3,4], ] ).T
        self.m = self.C.shape[1]
    
        super(TestDirectPattern, self).setup()
    
    def build_pattern(self):
        self.input_layer = lasagne.layers.InputLayer(shape=(None, self.n),
                                            input_var=self.input_var)
        self.phi = build_linear_simple( self.input_layer, self.m, name="phi")
        self.psi = build_linear_simple( self.phi, self.num_classes, 
            #nonlinearity=lasagne.nonlinearities.softmax,
            nonlinearity=None,
            name="psi")
            
        self.build_target_loss()
    
#        phi_output = self.phi.get_output_for(self.input_var)
#        psi_output = self.psi.get_output_for(phi_output).reshape((-1,))
        
        self.pattern = concarne.patterns.DirectPattern(phi=self.phi, psi=self.psi, 
                                             target_var=self.target_var, 
                                             context_var=self.context_var,
                                             target_loss=self.target_loss #lasagne.objectives.squared_error(
#                                                             psi_output,
#                                                             self.target_var).mean()
                                             )

    def test_pattern_output(self):
        assert (self.phi.W.get_value(borrow=True).shape == (2,1))
        self.phi.W.set_value ( np.array([[1,0]]).T )
        assert (self.phi.W.get_value(borrow=True).shape == (2,1))

        assert (self.psi.W.get_value(borrow=True).shape == (1,1))
        self.psi.W.set_value ( np.array([[1,]]) )
        assert (self.psi.W.get_value(borrow=True).shape == (1,1))
        
        test_prediction = lasagne.layers.get_output(self.pattern, deterministic=True)
        test_fn = theano.function([self.input_var], test_prediction)
        
        X_hat = test_fn(self.X)
    
        assert ( np.all(X_hat == self.S) )


    def test_pattern_training_loss_and_grads(self):
        loss = self.pattern.training_loss(**self.loss_weights).mean()
        train_fn_inputs = [self.input_var, self.target_var, self.context_var]
        
        train_fn = theano.function(train_fn_inputs, loss)
        assert (train_fn(self.X, self.Y, self.C) > 0)

        params = lasagne.layers.get_all_params(self.pattern, trainable=True)
        
        phi_params = lasagne.layers.get_all_params(self.phi, trainable=True)
        psi_params = lasagne.layers.get_all_params(self.psi, trainable=True)

        assert (len (params) == len(set (phi_params+psi_params)))
        
        lst = T.grad(loss, params)
        assert (len (lst) == len (params))


# ------------------------------------------------------------------------------

class TestMultiViewPattern(TestBase):
    """
      Test the multi view pattern with linear models
    """

    def setup(self):
        # embedded C
        self.C = np.array( [[0,1,2,3,4], [2,2,2,2,2]] ).T
        
        self.m = self.C.shape[1]
        self.d = 1

        super(TestMultiViewPattern, self).setup()
        
    def build_pattern(self):
        self.input_layer = lasagne.layers.InputLayer(shape=(None, self.n),
                                            input_var=self.input_var)
        self.context_input_layer = lasagne.layers.InputLayer(shape=(None, self.m),
                                            input_var=self.context_var)
        self.phi = build_linear_simple( self.input_layer, self.d, name="phi")
        self.psi = build_linear_simple( self.phi, self.num_classes, 
            nonlinearity=None, name="psi")
        self.beta = build_linear_simple( self.context_input_layer, self.d, name="beta")
            
        self.build_target_loss()
        
        self.pattern = concarne.patterns.MultiViewPattern(phi=self.phi, psi=self.psi, 
                                             beta=self.beta,
                                             target_var=self.target_var, 
                                             context_var=self.context_var,
                                             target_loss=self.target_loss,
                                             )

    def test_pattern_output(self):

        print (self.phi.W.get_value(borrow=True))
        assert (self.phi.W.get_value(borrow=True).shape == (self.n,self.d))
        self.phi.W.set_value ( np.array([[1,0]]).T )
        assert (self.phi.W.get_value(borrow=True).shape == (self.n,self.d))

        assert (self.psi.W.get_value(borrow=True).shape == (self.d,self.num_classes))
        self.psi.W.set_value ( np.array([[1,]]) )
        assert (self.psi.W.get_value(borrow=True).shape == (self.d,self.num_classes))

        assert (self.beta.W.get_value(borrow=True).shape == (self.m, self.d))
        self.beta.W.set_value ( np.array([[1,0]]).T )
        assert (self.beta.W.get_value(borrow=True).shape == (self.m, self.d))
        
        test_prediction = lasagne.layers.get_output(self.pattern, deterministic=True)
        test_fn = theano.function([self.input_var], test_prediction)
        X_hat = test_fn(self.X)
        assert ( np.all(X_hat == self.S) )

        beta_prediction = lasagne.layers.get_output(self.beta, deterministic=True)
        beta_fn = theano.function([self.context_var], beta_prediction)
        C_hat = beta_fn(self.C)
        assert ( np.all(C_hat == self.S) )


    def test_pattern_training_loss_and_grads(self):
        loss = self.pattern.training_loss(**self.loss_weights).mean()
        train_fn_inputs = [self.input_var, self.target_var, self.context_var]
        
        train_fn = theano.function(train_fn_inputs, loss)
        assert (train_fn(self.X, self.Y, self.C) > 0)

        params = lasagne.layers.get_all_params(self.pattern, trainable=True)
        
        phi_params = lasagne.layers.get_all_params(self.phi, trainable=True)
        psi_params = lasagne.layers.get_all_params(self.psi, trainable=True)
        beta_params = lasagne.layers.get_all_params(self.beta, trainable=True)

        assert (len (params) == len(set (phi_params+psi_params+beta_params)))
        
        lst = T.grad(loss, params)
        assert (len (lst) == len (params))

# ------------------------------------------------------------------------------
    
if __name__ == "__main__":
    td = TestDirectPattern()
    td.setup()
    td.test_pattern_training_loss_and_grads()
    
    #tmv = TestMultiViewPattern()