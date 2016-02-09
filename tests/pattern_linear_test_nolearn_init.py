# -*- coding: utf-8 -*-

#!/usr/bin/env python
#
# This is the the same test as pattern_linear_test, but here we use the
# nolearn style definition to build the functions

from __future__ import print_function

import concarne
import concarne.patterns
import concarne.iterators

import lasagne
import theano
import theano.tensor as T

import numpy as np


class TestPatternBase(object):
    """
      Setup class generating some simplistic direct data
    """
    
    def __init__(self):
        self.loss_weights = {'target_weight':0.9, 'side_weight':0.1}
    
        self.phi = None
        self.psi = None
        self.beta = None
    
    def setup(self):
        # generate input data
        self.X = np.array( [[0,1,2,3,4], [1,1,1,1,1]]).T
        # generate targets 
        self.Y = np.array( [0, 0, 1, 1, 1], dtype='int32' ).reshape( (-1,1) )

        # generate desired representation
        self.S = self.X[:,:1]
        
        # dim of input
        self.n = self.X.shape[1]
        # dim of output (dim=1, but 2 classes!)
        self.num_classes = 2

        self.init_variables()
        self.build_pattern()

        self.phi = self.pattern.phi
        self.psi = self.pattern.psi
        self.beta = self.pattern.beta

    def init_variables(self):
        self.input_var = T.matrix('inputs')
        self.side_var = T.matrix('contexts')
        # do regression
        #self.target_var = T.ivector('targets')
        #self.target_var = T.vector('targets') 
        self.target_var = T.matrix('targets') # otherwise dim mismatch for psi
        self.num_classes = 1 # regression -> dim matters, not classes

        self.phi = [
            (lasagne.layers.DenseLayer,
             {'num_units': concarne.patterns.Pattern.PHI_OUTPUT_SHAPE, 
             'nonlinearity': None, 'b':None})]
            
        self.psi = [
            (lasagne.layers.DenseLayer,
             {'num_units': concarne.patterns.Pattern.PSI_OUTPUT_SHAPE, 
             'nonlinearity': None, 'b':None})]

        self.beta = [
            (lasagne.layers.DenseLayer,
             {'num_units': concarne.patterns.Pattern.BETA_OUTPUT_SHAPE, 
             'nonlinearity': None, 'b':None})]
             
    def build_pattern(self):
        """Implemented by derived classes"""
        raise NotImplemented()


# ------------------------------------------------------------------------------
class TestSinglePatternBase(TestPatternBase):
    def _test_pattern_training_loss_and_grads(self):
        """ A generic test method usable for verifying that the 
          test and train theano functions can be correctly built, and that
          the gradient for the parameters is computable.
          
          Needs to be called explicitly by subclasses because inheriting
          test cases with nosetests is problematic (and not very transparent)"""
          
        loss = self.pattern.training_loss(**self.loss_weights).mean()
        train_fn_inputs = [self.input_var, self.target_var, self.side_var]
        
        train_fn = theano.function(train_fn_inputs, loss)
        assert (train_fn(self.X, self.Y, self.C) > 0)

        params = lasagne.layers.get_all_params(self.pattern, trainable=True)
        
        phi_params = lasagne.layers.get_all_params(self.phi, trainable=True)
        psi_params = lasagne.layers.get_all_params(self.psi, trainable=True)
        if self.beta is not None:
            beta_params = lasagne.layers.get_all_params(self.beta, trainable=True)
        else:
            beta_params = []

        assert (len (params) == len(set (phi_params+psi_params+beta_params)))
        
        lst = T.grad(loss, params)
        assert (len (lst) == len (params))
        

    def _test_learn(self):
        loss = self.pattern.training_loss(**self.loss_weights).mean()
        train_fn_inputs = [self.input_var, self.target_var, self.side_var]
        
        params = lasagne.layers.get_all_params(self.pattern, trainable=True)
        
        updates = lasagne.updates.nesterov_momentum(
                loss, params, learning_rate=1e-5, momentum=0.9)
        
        train_fn = theano.function(train_fn_inputs, loss, updates=updates)
        #assert (train_fn(self.X, self.Y, self.C) > 0)
        
#        test_prediction = lasagne.layers.get_output(self.pattern, deterministic=True)
#        test_loss = lasagne.objectives.categorical_crossentropy(test_prediction,
#                                                                self.target_var)
#        test_loss = test_loss.mean()
#        # As a bonus, also create an expression for the classification accuracy:
##        test_acc = T.mean(T.eq(T.argmax(test_prediction, axis=1), self.target_var),
##                          dtype=theano.config.floatX)
#        test_fn = theano.function([self.input_var, self.target_var], test_loss)
        
        num_epochs = 1000
        for epoch in range(num_epochs):
            train_err = 0
            train_batches = 0
            sit = concarne.iterators.AlignedBatchIterator(2)
            for X, Y, C in sit(self.X, self.Y, self.C):
                train_err += train_fn(X,Y,C)
                train_batches += 1

        # we cannot guarantee any training or test error here
        # we are simply happy if the iteration didn't crash

#            print("Epoch {} of {}".format(
#                epoch + 1, num_epochs, ))
#            print("  training loss:\t\t{:.6f}".format(train_err / train_batches))
#        
#        assert (train_err < 1e-5)

        
# ------------------------------------------------------------------------------

class TestDirectPattern(TestSinglePatternBase):
    """
      Test the direct pattern with linear models
    """

    def setup(self):
        # define direct side info data
        self.C = np.array( [[0,1,2,3,4], ] ).T
        self.m = self.C.shape[1]
    
        super(TestDirectPattern, self).setup()
    
    def build_pattern(self):
#        self.phi = [
#            (lasagne.layers.DenseLayer,
#             {'num_units': self.m, 'nonlinearity': None, 'b':None})]
#            
#        self.psi = [
#            (lasagne.layers.DenseLayer,
#             {'num_units': self.num_classes, 'nonlinearity': None, 'b':None})]

        self.pattern = concarne.patterns.DirectPattern(phi=self.phi, psi=self.psi, 
                                             input_var=self.input_var,
                                             target_var=self.target_var, 
                                             side_var=self.side_var,
                                             input_shape=self.n,
                                             side_shape=self.m,
                                             target_shape=self.num_classes,
                                             representation_shape=self.m,
                                             target_loss=lasagne.objectives.squared_error,
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
        self._test_pattern_training_loss_and_grads()
        
    def test_learn(self):
        self._test_learn()

# ------------------------------------------------------------------------------

class TestMultiViewPattern(TestSinglePatternBase):
    """
      Test the multi view pattern with linear models
    """

    def setup(self):
        # define embedded C
        self.C = np.array( [[0,1,2,3,4], [2,2,2,2,2]] ).T
        
        # dim of side info
        self.m = self.C.shape[1]
        
        # size of desired intermediate representation
        self.d = 1

        super(TestMultiViewPattern, self).setup()
        
    def build_pattern(self):
        self.pattern = concarne.patterns.MultiViewPattern(phi=self.phi, psi=self.psi, 
                                             beta=self.beta,
                                             input_var=self.input_var,
                                             target_var=self.target_var, 
                                             side_var=self.side_var,
                                             input_shape=self.n,
                                             side_shape=self.m,
                                             target_shape=self.num_classes,
                                             representation_shape=self.d,
                                             target_loss=lasagne.objectives.squared_error,
                                             )        
        
    def test_pattern_output(self):
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
        beta_fn = theano.function([self.side_var], beta_prediction)
        C_hat = beta_fn(self.C)
        assert ( np.all(C_hat == self.S) )

    def test_pattern_training_loss_and_grads(self):
        self._test_pattern_training_loss_and_grads()

    def test_learn(self):
        self._test_learn()

# ------------------------------------------------------------------------------

class TestMultiTaskPattern(TestSinglePatternBase):
    """
      Test the multi task pattern with linear models
    """

    def setup(self):
        # define embedded C
        self.C = np.array( [[0,1,2,3,4], [2,2,2,2,2]] ).T
        
        # dim of side info
        self.m = self.C.shape[1]
        
        # size of desired intermediate representation
        self.d = 1

        super(TestMultiTaskPattern, self).setup()
        
    def build_pattern(self):
#        self.phi = [
#            (lasagne.layers.DenseLayer,
#             {'num_units': self.d, 'nonlinearity': None, 'b':None})]
#            
#        self.psi = [
#            (lasagne.layers.DenseLayer,
#             {'num_units': self.num_classes, 'nonlinearity': None, 'b':None})]
#
#        self.beta = [
#            (lasagne.layers.DenseLayer,
#             {'num_units': self.m, 'nonlinearity': None, 'b':None})]

        self.pattern = concarne.patterns.MultiTaskPattern(phi=self.phi, psi=self.psi, 
                                             beta=self.beta,
                                             input_var=self.input_var,
                                             target_var=self.target_var, 
                                             side_var=self.side_var,
                                             input_shape=self.n,
                                             side_shape=self.m,
                                             target_shape=self.num_classes,
                                             representation_shape=self.d,
                                             target_loss=lasagne.objectives.squared_error,
                                             )

    def test_pattern_output(self):
        assert (self.phi.W.get_value(borrow=True).shape == (self.n,self.d))
        self.phi.W.set_value ( np.array([[1,0]]).T )
        assert (self.phi.W.get_value(borrow=True).shape == (self.n,self.d))

        assert (self.psi.W.get_value(borrow=True).shape == (self.d,self.num_classes))
        self.psi.W.set_value ( np.array([[1,]]) )
        assert (self.psi.W.get_value(borrow=True).shape == (self.d,self.num_classes))

        assert (self.beta.W.get_value(borrow=True).shape == (self.d, self.m))
        # [1,1] means that we will project the intermediate representation
        # onto both dimensions of the output representation
        self.beta.W.set_value ( np.array([[1,1]]) )
        assert (self.beta.W.get_value(borrow=True).shape == (self.d, self.m))
        
        test_prediction = lasagne.layers.get_output(self.pattern, deterministic=True)
        test_fn = theano.function([self.input_var], test_prediction)
        X_hat = test_fn(self.X)
        assert ( np.all(X_hat == self.S) )

        beta_prediction = lasagne.layers.get_output(self.beta, deterministic=True)
        beta_fn = theano.function([self.input_var], beta_prediction)
        C_hat = beta_fn(self.X)
        assert ( np.all(C_hat[:,0] == self.S[:,0]) )
        assert ( np.all(C_hat[:,1] == self.S[:,0]) )

    def test_pattern_training_loss_and_grads(self):
        self._test_pattern_training_loss_and_grads()

    def test_learn(self):
        self._test_learn()


# ------------------------------------------------------------------------------

class TestPWTransformationPattern(TestPatternBase):
    """
      Test the pairwise transformation pattern with linear models
    """

    def setup(self):
        # define embedded C
        self.CX = np.array( [[1,2,3,4,5], [2,2,2,2,2]] ).T
        self.Cy = -np.array( [[1,1,1,1,1]] ).T
        
        # dim of side info input == dim of X
        self.m = self.CX.shape[1]
        
        # size of desired intermediate representation
        self.d = 1

        self.my = self.Cy.shape[1]

        super(TestPWTransformationPattern, self).setup()
        
        assert(self.CX.shape[1] == self.X.shape[1])
        
    def build_pattern(self):
        self.side_transform_var = T.matrix('side_transforms')

#        self.phi = [
#            (lasagne.layers.DenseLayer,
#             {'num_units': self.d, 'nonlinearity': None, 'b':None})]
#            
#        self.psi = [
#            (lasagne.layers.DenseLayer,
#             {'num_units': self.num_classes, 'nonlinearity': None, 'b':None})]

        self.pattern = concarne.patterns.PairwisePredictTransformationPattern(
                                             phi=self.phi, psi=self.psi, 
                                             input_var=self.input_var,
                                             target_var=self.target_var, 
                                             side_var=self.side_var,
                                             side_transform_var=self.side_transform_var,
                                             input_shape=self.n,
                                             side_shape=self.m,
                                             target_shape=self.num_classes,
                                             representation_shape=self.d,
                                             side_transform_shape=self.my,
                                             target_loss=lasagne.objectives.squared_error,
                                             )        

    def test_pattern_output(self):
        #print (self.phi.W.get_value(borrow=True))
        assert (self.phi.W.get_value(borrow=True).shape == (self.n,self.d))
        self.phi.W.set_value ( np.array([[1,0]]).T )
        assert (self.phi.W.get_value(borrow=True).shape == (self.n,self.d))

        assert (self.psi.W.get_value(borrow=True).shape == (self.d,self.num_classes))
        self.psi.W.set_value ( np.array([[1,]]) )
        assert (self.psi.W.get_value(borrow=True).shape == (self.d,self.num_classes))

#        assert (self.beta.W.get_value(borrow=True).shape == (self.d, self.m))
#        # [1,1] means that we will project the intermediate representation
#        # onto both dimensions of the output representation
#        self.beta.W.set_value ( np.array([[1,1]]) )
#        assert (self.beta.W.get_value(borrow=True).shape == (self.d, self.m))
        
        test_prediction = lasagne.layers.get_output(self.pattern, deterministic=True)
        test_fn = theano.function([self.input_var], test_prediction)
        X_hat = test_fn(self.X)
        assert ( np.all(X_hat == self.S) )
        
        
#        self.phi1 = test_prediction
#        self.phi2 = lasagne.layers.get_output(self.pattern, self.side_var, deterministic=True)
        beta_prediction = self.pattern.get_beta_output_for(self.input_var, self.side_var, deterministic=True)
        beta_fn = theano.function([self.input_var, self.side_var], beta_prediction)
        C_hat = beta_fn(self.X, self.CX)
        assert ( np.all(C_hat == self.Cy) )

    def test_pattern_training_loss_and_grads(self):
        loss = self.pattern.training_loss(**self.loss_weights).mean()
        train_fn_inputs = [self.input_var, self.target_var, self.side_var, self.side_transform_var]
        
        train_fn = theano.function(train_fn_inputs, loss)
        assert (train_fn(self.X, self.Y, self.CX, self.Cy) > 0)

        params = lasagne.layers.get_all_params(self.pattern, trainable=True)
        
        phi_params = lasagne.layers.get_all_params(self.phi, trainable=True)
        psi_params = lasagne.layers.get_all_params(self.psi, trainable=True)
        if self.beta is not None:
            beta_params = lasagne.layers.get_all_params(self.beta, trainable=True)
        else:
            beta_params = []

        assert (len (params) == len(set (phi_params+psi_params+beta_params)))
        
        lst = T.grad(loss, params)
        assert (len (lst) == len (params))


    def test_learn(self):
        loss = self.pattern.training_loss(**self.loss_weights).mean()
        train_fn_inputs = [self.input_var, self.target_var, self.side_var, self.side_transform_var]
        
        params = lasagne.layers.get_all_params(self.pattern, trainable=True)
        
        updates = lasagne.updates.nesterov_momentum(
                loss, params, learning_rate=1e-5, momentum=0.9)
        
        train_fn = theano.function(train_fn_inputs, loss, updates=updates)
        
        num_epochs = 1000
        for epoch in range(num_epochs):
            train_err = 0
            train_batches = 0
            sit = concarne.iterators.AlignedBatchIterator(2)
            for X, Y, CX, Cy in sit(self.X, self.Y, self.CX, self.Cy):
                train_err += train_fn(X,Y,CX,Cy)
                train_batches += 1
                
                
# ------------------------------------------------------------------------------
if __name__ == "__main__":
    td = TestDirectPattern()
    #td = TestMultiViewPattern()
    #td = TestMultiTaskPattern()
    #td = TestPWTransformationPattern()
    td.setup()
    td.test_pattern_output()
    td.test_pattern_training_loss_and_grads()
    td.test_learn()
