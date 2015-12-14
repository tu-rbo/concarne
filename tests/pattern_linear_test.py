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


class TestDirectData(object):
    """
      Setup class generating some simplistic direct data
    """
    def setup(self):
        self.X = np.array( [[0,1,2,3,4], [1,1,1,1,1]]).T
        self.S = self.X[:,:1]
        self.Y = np.array( [0, 0, 1, 1, 1], dtype='int32' )
        self.C = np.array( [[0,1,2,3,4], ] ).T
        
        self.n = self.X.shape[1]
        self.m = self.C.shape[1]
        
        self.num_classes = 2

class TestDirectPattern(TestDirectData):
    """
      Test the direct pattern with linear models
    """
    

    def setup(self):
        super(TestDirectPattern, self).setup()

        self.init_variables()
        self.build_pattern()
        
    def init_variables(self):
        self.input_var = T.matrix('inputs')
        #self.target_var = T.ivector('targets')
        # do regression
        self.target_var = T.vector('targets')
        self.context_var = T.matrix('contexts')

        self.num_classes = 1

    def build_pattern(self):

        self.input_layer = lasagne.layers.InputLayer(shape=(None, self.n),
                                            input_var=self.input_var)
        self.phi = build_linear_simple( self.input_layer, self.m, name="phi")
        self.psi = build_linear_simple( self.phi, self.num_classes, 
            #nonlinearity=lasagne.nonlinearities.softmax,
            nonlinearity=None,
            name="psi")
    
        phi_output = self.phi.get_output_for(self.input_var)
        psi_output = self.psi.get_output_for(phi_output).reshape((-1,))
        
        self.pattern = concarne.patterns.DirectPattern(phi=self.phi, psi=self.psi, 
                                             target_var=self.target_var, 
                                             context_var=self.context_var,
                                             target_loss=lasagne.objectives.squared_error(
                                                             psi_output,
                                                             self.target_var).mean()
                                             )

        

    def test_pattern_output(self):

        assert (self.phi.W.get_value(borrow=True).shape == (2,1))
        self.phi.W.set_value ( np.array([[1,0]]).T )
        assert (self.phi.W.get_value(borrow=True).shape == (2,1))

        #assert (psi.W.get_value(borrow=True).shape == (2,1))
        assert (self.psi.W.get_value(borrow=True).shape == (1,1))
        self.psi.W.set_value ( np.array([[1,]]) )
        
        test_prediction = lasagne.layers.get_output(self.pattern, deterministic=True)
        test_fn = theano.function([self.input_var], test_prediction)
        
        X_hat = test_fn(self.X)
    
        assert ( np.all(X_hat == self.S) )


    def test_pattern_training_loss_and_grads(self):
        
        loss_weights = {'target_weight':0.9, 'context_weight':0.1}
        loss = self.pattern.training_loss(**loss_weights).mean()
        train_fn_inputs = [self.input_var, self.target_var, self.context_var]
        
        train_fn = theano.function(train_fn_inputs, loss)
        assert (train_fn(self.X, self.Y, self.C) > 0)

        params = lasagne.layers.get_all_params(self.pattern, trainable=True)
        
        phi_params = lasagne.layers.get_all_params(self.phi, trainable=True)
        psi_params = lasagne.layers.get_all_params(self.psi, trainable=True)

        assert (len (params) == len(set (phi_params+(psi_params))))
        
        lst = T.grad(loss, params)
        assert (len (lst) == len (params))


# ------------------------------------------------------------------------------

class TestMultiViewPattern(TestDirectData):
    """
      Test the multi view pattern with linear models
    """

    def setup(self):
        super(TestMultiViewPattern, self).setup()
        # embedded C
        self.C = np.array( [[0,1,2,3,4], [2,2,2,2,2]] ).T


#    def test_pattern(self):
#        input_var = T.matrix('inputs')
#        target_var = T.ivector('targets')
#        context_var = T.matrix('contexts')
#        
#        input_layer = lasagne.layers.InputLayer(shape=(None, self.n),
#                                            input_var=input_var)
#        phi = build_linear_simple( input_layer, self.m, name="phi")
#        psi = build_linear_simple( phi, self.num_classes, 
#            #nonlinearity=lasagne.nonlinearities.softmax,
#            nonlinearity=None,
#            name="psi")
#    
#        dp = concarne.patterns.DirectPattern(phi=phi, psi=psi, 
#                                             target_var=target_var, 
#                                             context_var=context_var,
#                                             target_loss=lasagne.objectives.squared_error(
#                                                             lasagne.layers.get_output(phi, input_var),
#                                                             #phi.get_output_for(input_var), 
#                                                             context_var)
#                                             )
#                                      
#        assert (phi.W.get_value(borrow=True).shape == (2,1))
#        phi.W.set_value ( np.array([[1,0]]).T )
#        assert (phi.W.get_value(borrow=True).shape == (2,1))
#
#        #assert (psi.W.get_value(borrow=True).shape == (2,1))
#        assert (psi.W.get_value(borrow=True).shape == (1,1))
#        psi.W.set_value ( np.array([[1,]]) )
#        
#        test_prediction = lasagne.layers.get_output(dp, deterministic=True)
#        test_fn = theano.function([input_var], test_prediction)
#        
#        X_hat = test_fn(self.X)
#    
#        assert ( np.all(self.C == X_hat) )

    
if __name__ == "__main__":
    td = TestDirectPattern()
    td.setup()
    td.test_pattern_training_loss_and_grads()
    
    #tmv = TestMultiViewPattern()