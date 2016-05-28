# -*- coding: utf-8 -*-

#!/usr/bin/env python

from __future__ import print_function

import concarne
import concarne.patterns
import concarne.training
import concarne.iterators

import lasagne
import theano
import theano.tensor as T

import numpy as np


def build_linear_simple(input_layer, n_out, nonlinearity=None, name=None):
    network = lasagne.layers.DenseLayer(input_layer, n_out, nonlinearity=nonlinearity, b=None, name=name)
    return network    


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
        # generate targets (FIXME currently unused)
        self.Y = np.array( [0, 0, 1, 1, 1], dtype='int32' )

        # validation data
        self.X_val = np.array( [[0.5,1.5,2.5,3.5], [1,1,1,1]]).T
        self.Y_val = np.array( [0, 0, 1, 1,], dtype='int32' )

        # generate desired representation
        self.S = self.X[:,:1]
        
        # dim of input
        self.n = self.X.shape[1]
        # dim of output (dim=1, but 2 classes!)
        self.num_classes = 2

        self.init_variables()
        self.build_pattern()

    def init_variables(self):
        self.input_var = T.matrix('inputs')
        self.side_var = T.matrix('contexts')
        # do regression
        #self.target_var = T.ivector('targets')
        self.target_var = T.vector('targets')
        self.num_classes = 1 # regression -> dim matters, not classes

    def build_pattern(self):
        """Implemented by derived classes"""
        raise NotImplemented()

    # deprecated:
#     def build_target_loss(self):
#         phi_output = self.phi.get_output_for(self.input_var)
#         psi_output = self.psi.get_output_for(phi_output).reshape((-1,))
#         self.target_loss = lasagne.objectives.squared_error(psi_output, self.target_var).mean()

    def build_target_loss(self):
        self.target_loss = lasagne.objectives.squared_error

    def build_and_run_pattern_trainer(self, procedure):
        optional_kwargs = {}
        
        # to avoid warnings
        if procedure == "pretrain_finetune":
            optional_kwargs ['XYpsi_num_epochs'] = 1
        if procedure != "simultaneous":
            optional_kwargs ['XZ_num_epochs'] = 1
    
        self.trainer = concarne.training.PatternTrainer(self.pattern,
                                               procedure,
                                               num_epochs=1,
                                               batch_size=2,
                                               update=lasagne.updates.nesterov_momentum,
                                               update_learning_rate=1e-5,
                                               update_momentum=0.9,
                                               save_params=False,
                                               side_weight=0.5,
                                               target_weight=0.5,
                                               **optional_kwargs
                                               )
        return self.trainer
        
        
    def _test_pattern_trainer_predict(self, verbose=False):
        self.build_and_run_pattern_trainer(procedure='simultaneous')
        # just check it doesn't crash
        res = self.trainer.predict(self.X)
        assert (not np.any(np.isnan(res)))

    def _test_pattern_trainer_predict_proba(self, verbose=False):
        self.build_and_run_pattern_trainer(procedure='simultaneous')
        # just check it doesn't crash and is not nan
        res = self.trainer.predict_proba(self.X)
        assert (not np.any(np.isnan(res)))


# ------------------------------------------------------------------------------
class TestSinglePatternBase(TestPatternBase):
    def _test_pattern_training_loss_and_grads(self):
        """ A generic test method usable for verifying that the 
          test and train theano functions can be correctly built, and that
          the gradient for the parameters is computable.
          
          Needs to be called explicitly by subclasses because inheriting
          test cases with nosetests is problematic (and not very transparent)"""
          
        loss, tloss, sloss = self.pattern.training_loss(all_losses=True, **self.loss_weights)
        loss = loss.mean()
        train_fn_inputs = [self.input_var, self.target_var, self.side_var]
        
        train_fn = theano.function(train_fn_inputs, [loss,tloss, sloss])
        l, tl, sl = train_fn(self.X, self.Y, self.C)
#         assert (tl > 0)
#         assert (sl > 0)
        assert (l > 0)

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

    def _test_pattern_trainer_XYZ_no_val(self, procedure='simultaneous',verbose=False):
        self.build_and_run_pattern_trainer(procedure)
        self.trainer.fit_XYZ(self.X, self.Y, [self.C], verbose=verbose)
        
        return self.trainer.score(self.X, self.Y)
        
    def _test_pattern_trainer_XYZ_valXY(self, procedure='simultaneous', verbose=False):
        self.build_and_run_pattern_trainer(procedure)
        self.trainer.fit_XYZ(self.X, self.Y, [self.C], X_val=self.X_val, y_val=self.Y_val,
            verbose=verbose)
        
        return self.trainer.score(self.X_val, self.Y_val)

    def _test_pattern_trainer_XYZ_valXYZ(self, procedure='simultaneous', verbose=False):
        self.build_and_run_pattern_trainer(procedure)
        self.trainer.fit_XYZ(self.X, self.Y, [self.C], X_val=self.X_val, y_val=self.Y_val,
            side_val=[self.X_val, self.C_val], verbose=verbose)
        
        res = []
        res += self.trainer.score(self.X_val, self.Y_val)
        res += self.trainer.score_side([self.X_val, self.C_val])
        return res
         
    def _test_pattern_trainer_XZ_XY_valXYZ (self, procedure='simultaneous', verbose=False):
        self.build_and_run_pattern_trainer(procedure)
        # different length
        XZ = [self.X[1:], self.C[1:]]
        XY = [self.X[:-1], self.Y[1:]]
        
        self.trainer.fit_XZ_XY(XZ[0], [XZ[1]], XY[0], XY[1],
            X_val=self.X_val, y_val=self.Y_val,
            side_val=[self.X_val, self.C_val], verbose=verbose)
        
        res = []
        res += self.trainer.score(self.X_val, self.Y_val)
        res += self.trainer.score_side([self.X_val, self.C_val])
        
        return res
        
# ------------------------------------------------------------------------------

class TestDirectPattern(TestSinglePatternBase):
    """
      Test the direct pattern with linear models
    """

    def setup(self):
        # define direct side data
        self.C = np.array( [[0,1,2,3,4], ] ).T
        self.m = self.C.shape[1]
        
        self.C_val = np.array( [[0.5,1.5,2.5,3.5], ] ).T
    
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
        
        self.pattern = concarne.patterns.DirectPattern(phi=self.phi, psi=self.psi, 
                                             target_var=self.target_var, 
                                             side_var=self.side_var,
                                             target_loss=self.target_loss
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

    #---
    def test_pattern_trainer_XYZ_simul_no_val(self, verbose=False):
        res = self._test_pattern_trainer_XYZ_no_val(procedure="simultaneous", verbose=verbose)
        assert(not np.isnan(res[0]))
        assert(np.isnan(res[1])) # because MSE

    def test_pattern_trainer_XYZ_decoupled_no_val(self, verbose=False):
        res = self._test_pattern_trainer_XYZ_no_val(procedure="decoupled", verbose=verbose)
        assert(not np.isnan(res[0]))
        assert(np.isnan(res[1])) # because MSE

    def test_pattern_trainer_XYZ_prefine_no_val(self, verbose=False):
        res = self._test_pattern_trainer_XYZ_no_val(procedure="pretrain_finetune", verbose=verbose)
        assert(not np.isnan(res[0]))
        assert(np.isnan(res[1])) # because MSE

    #---
    def test_pattern_trainer_XYZ_simul_valXY(self, verbose=False):
        res = self._test_pattern_trainer_XYZ_valXY(procedure="simultaneous", verbose=verbose)
        assert(not np.isnan(res[0]))
        assert(np.isnan(res[1])) # because MSE

    def test_pattern_trainer_XYZ_decoupled_valXY(self, verbose=False):
        res = self._test_pattern_trainer_XYZ_valXY(procedure="decoupled", verbose=verbose)
        assert(not np.isnan(res[0]))
        assert(np.isnan(res[1])) # because MSE

    def test_pattern_trainer_XYZ_prefine_valXY(self, verbose=False):
        res = self._test_pattern_trainer_XYZ_valXY(procedure="pretrain_finetune", verbose=verbose)
        assert(not np.isnan(res[0]))
        assert(np.isnan(res[1])) # because MSE

    #---
    def test_pattern_trainer_XYZ_simul_valXYZ(self, verbose=False):
        res = self._test_pattern_trainer_XYZ_valXYZ(procedure="simultaneous", verbose=verbose)
        assert(not np.isnan(res[0]))
        assert(np.isnan(res[1])) # because MSE
        assert(not np.isnan(res[2]))
        assert(np.isnan(res[3])) # because MSE

    def test_pattern_trainer_XYZ_decoupled_valXYZ(self, verbose=False):
        res = self._test_pattern_trainer_XYZ_valXYZ(procedure="decoupled", verbose=verbose)
        assert(not np.isnan(res[0]))
        assert(np.isnan(res[1])) # because MSE
        assert(not np.isnan(res[2]))
        assert(np.isnan(res[3])) # because MSE

    def test_pattern_trainer_XYZ_prefine_valXYZ(self, verbose=False):
        res = self._test_pattern_trainer_XYZ_valXYZ(procedure="pretrain_finetune", verbose=verbose)
        assert(not np.isnan(res[0]))
        assert(np.isnan(res[1])) # because MSE
        assert(not np.isnan(res[2]))
        assert(np.isnan(res[3])) # because MSE

    #---
    def test_pattern_trainer_XZ_XY_simul_valXZ_XY(self, verbose=False):
        res = self._test_pattern_trainer_XZ_XY_valXYZ(procedure="simultaneous", verbose=verbose)
        assert(not np.isnan(res[0]))
        assert(np.isnan(res[1])) # because MSE
        assert(not np.isnan(res[2]))
        assert(np.isnan(res[3])) # because MSE

    def test_pattern_trainer_XZ_XY_decoupled_valXZ_XY(self, verbose=False):
        res = self._test_pattern_trainer_XZ_XY_valXYZ(procedure="decoupled", verbose=verbose)
        assert(not np.isnan(res[0]))
        assert(np.isnan(res[1])) # because MSE
        assert(not np.isnan(res[2]))
        assert(np.isnan(res[3])) # because MSE

    def test_pattern_trainer_XZ_XY_prefine_valXZ_XY(self, verbose=False):
        res = self._test_pattern_trainer_XZ_XY_valXYZ(procedure="pretrain_finetune", verbose=verbose)
        assert(not np.isnan(res[0]))
        assert(np.isnan(res[1])) # because MSE
        assert(not np.isnan(res[2]))
        assert(np.isnan(res[3])) # because MSE

    def test_pattern_trainer_predict(self):
        self._test_pattern_trainer_predict()

# ------------------------------------------------------------------------------

class TestMultiViewPattern(TestSinglePatternBase):
    """
      Test the multi view pattern with linear models
    """

    def setup(self):
        # define embedded C
        self.C = np.array( [[0,1,2,3,4], [2,2,2,2,2]] ).T
        self.C_val = np.array( [[0.5,1.5,2.5,3.5], [2,2,2,2] ] ).T
        
        # dim of side info
        self.m = self.C.shape[1]
        
        # size of desired intermediate representation
        self.d = 1

        super(TestMultiViewPattern, self).setup()
        
    def build_pattern(self):
        self.input_layer = lasagne.layers.InputLayer(shape=(None, self.n),
                                            input_var=self.input_var)
        self.side_input_layer = lasagne.layers.InputLayer(shape=(None, self.m),
                                            input_var=self.side_var)
        self.phi = build_linear_simple( self.input_layer, self.d, name="phi")
        self.psi = build_linear_simple( self.phi, self.num_classes, 
            nonlinearity=None, name="psi")
        self.beta = build_linear_simple( self.side_input_layer, self.d, name="beta")
            
        self.build_target_loss()
        
        self.pattern = concarne.patterns.MultiViewPattern(phi=self.phi, psi=self.psi, 
                                             beta=self.beta,
                                             target_var=self.target_var, 
                                             side_var=self.side_var,
                                             target_loss=self.target_loss,
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

    #---
    def test_pattern_trainer_XYZ_simul_no_val(self, verbose=False):
        res = self._test_pattern_trainer_XYZ_no_val(procedure="simultaneous", verbose=verbose)
        assert(not np.isnan(res[0]))
        assert(np.isnan(res[1])) # because MSE

    def test_pattern_trainer_XYZ_decoupled_no_val(self, verbose=False):
        res = self._test_pattern_trainer_XYZ_no_val(procedure="decoupled", verbose=verbose)
        assert(not np.isnan(res[0]))
        assert(np.isnan(res[1])) # because MSE

    def test_pattern_trainer_XYZ_prefine_no_val(self, verbose=False):
        res = self._test_pattern_trainer_XYZ_no_val(procedure="pretrain_finetune", verbose=verbose)
        assert(not np.isnan(res[0]))
        assert(np.isnan(res[1])) # because MSE

    #---
    def test_pattern_trainer_XYZ_simul_valXY(self, verbose=False):
        res = self._test_pattern_trainer_XYZ_valXY(procedure="simultaneous", verbose=verbose)
        assert(not np.isnan(res[0]))
        assert(np.isnan(res[1])) # because MSE

    def test_pattern_trainer_XYZ_decoupled_valXY(self, verbose=False):
        res = self._test_pattern_trainer_XYZ_valXY(procedure="decoupled", verbose=verbose)
        assert(not np.isnan(res[0]))
        assert(np.isnan(res[1])) # because MSE

    def test_pattern_trainer_XYZ_prefine_valXY(self, verbose=False):
        res = self._test_pattern_trainer_XYZ_valXY(procedure="pretrain_finetune", verbose=verbose)
        assert(not np.isnan(res[0]))
        assert(np.isnan(res[1])) # because MSE

    #---
    def test_pattern_trainer_XYZ_simul_valXYZ(self, verbose=False):
        res = self._test_pattern_trainer_XYZ_valXYZ(procedure="simultaneous", verbose=verbose)
        assert(not np.isnan(res[0]))
        assert(np.isnan(res[1])) # because MSE
        assert(not np.isnan(res[2]))
        assert(np.isnan(res[3])) # because MSE

    def test_pattern_trainer_XYZ_decoupled_valXYZ(self, verbose=False):
        res = self._test_pattern_trainer_XYZ_valXYZ(procedure="decoupled", verbose=verbose)
        assert(not np.isnan(res[0]))
        assert(np.isnan(res[1])) # because MSE
        assert(not np.isnan(res[2]))
        assert(np.isnan(res[3])) # because MSE

    def test_pattern_trainer_XYZ_prefine_valXYZ(self, verbose=False):
        res = self._test_pattern_trainer_XYZ_valXYZ(procedure="pretrain_finetune", verbose=verbose)
        assert(not np.isnan(res[0]))
        assert(np.isnan(res[1])) # because MSE
        assert(not np.isnan(res[2]))
        assert(np.isnan(res[3])) # because MSE

    #---
    def test_pattern_trainer_XZ_XY_simul_valXZ_XY(self, verbose=False):
        res = self._test_pattern_trainer_XZ_XY_valXYZ(procedure="simultaneous", verbose=verbose)
        assert(not np.isnan(res[0]))
        assert(np.isnan(res[1])) # because MSE
        assert(not np.isnan(res[2]))
        assert(np.isnan(res[3])) # because MSE

    def test_pattern_trainer_XZ_XY_decoupled_valXZ_XY(self, verbose=False):
        res = self._test_pattern_trainer_XZ_XY_valXYZ(procedure="decoupled", verbose=verbose)
        assert(not np.isnan(res[0]))
        assert(np.isnan(res[1])) # because MSE
        assert(not np.isnan(res[2]))
        assert(np.isnan(res[3])) # because MSE

    def test_pattern_trainer_XZ_XY_prefine_valXZ_XY(self, verbose=False):
        res = self._test_pattern_trainer_XZ_XY_valXYZ(procedure="pretrain_finetune", verbose=verbose)
        assert(not np.isnan(res[0]))
        assert(np.isnan(res[1])) # because MSE
        assert(not np.isnan(res[2]))
        assert(np.isnan(res[3])) # because MSE

    def test_pattern_trainer_predict(self):
        self._test_pattern_trainer_predict()

# ------------------------------------------------------------------------------

class TestMultiTaskPattern(TestSinglePatternBase):
    """
      Test the multi task pattern with linear models
    """

    def setup(self):
        # define embedded C
        self.C = np.array( [[0,1,2,3,4], [2,2,2,2,2]] ).T

        self.C_val = np.array( [[0.5,1.5,2.5,3.5], [2,2,2,2]] ).T
        
        # dim of side info
        self.m = self.C.shape[1]
        
        # size of desired intermediate representation
        self.d = 1

        super(TestMultiTaskPattern, self).setup()
        
    def build_pattern(self):
        self.input_layer = lasagne.layers.InputLayer(shape=(None, self.n),
                                            input_var=self.input_var)
        self.phi = build_linear_simple( self.input_layer, self.d, name="phi")
        self.psi = build_linear_simple( self.phi, self.num_classes, 
            nonlinearity=None, name="psi")
        self.beta = build_linear_simple( self.phi, self.m, name="beta")
            
        self.build_target_loss()
        
        self.pattern = concarne.patterns.MultiTaskPattern(phi=self.phi, psi=self.psi, 
                                             beta=self.beta,
                                             target_var=self.target_var, 
                                             side_var=self.side_var,
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

    #---
    def test_pattern_trainer_XYZ_simul_no_val(self, verbose=False):
        res = self._test_pattern_trainer_XYZ_no_val(procedure="simultaneous", verbose=verbose)
        assert(not np.isnan(res[0]))
        assert(np.isnan(res[1])) # because MSE

    def test_pattern_trainer_XYZ_decoupled_no_val(self, verbose=False):
        res = self._test_pattern_trainer_XYZ_no_val(procedure="decoupled", verbose=verbose)
        assert(not np.isnan(res[0]))
        assert(np.isnan(res[1])) # because MSE

    def test_pattern_trainer_XYZ_prefine_no_val(self, verbose=False):
        res = self._test_pattern_trainer_XYZ_no_val(procedure="pretrain_finetune", verbose=verbose)
        assert(not np.isnan(res[0]))
        assert(np.isnan(res[1])) # because MSE

    #---
    def test_pattern_trainer_XYZ_simul_valXY(self, verbose=False):
        res = self._test_pattern_trainer_XYZ_valXY(procedure="simultaneous", verbose=verbose)
        assert(not np.isnan(res[0]))
        assert(np.isnan(res[1])) # because MSE

    def test_pattern_trainer_XYZ_decoupled_valXY(self, verbose=False):
        res = self._test_pattern_trainer_XYZ_valXY(procedure="decoupled", verbose=verbose)
        assert(not np.isnan(res[0]))
        assert(np.isnan(res[1])) # because MSE

    def test_pattern_trainer_XYZ_prefine_valXY(self, verbose=False):
        res = self._test_pattern_trainer_XYZ_valXY(procedure="pretrain_finetune", verbose=verbose)
        assert(not np.isnan(res[0]))
        assert(np.isnan(res[1])) # because MSE

    #---
    def test_pattern_trainer_XYZ_simul_valXYZ(self, verbose=False):
        res = self._test_pattern_trainer_XYZ_valXYZ(procedure="simultaneous", verbose=verbose)
        assert(not np.isnan(res[0]))
        assert(np.isnan(res[1])) # because MSE
        assert(not np.isnan(res[2]))
        assert(np.isnan(res[3])) # because MSE

    def test_pattern_trainer_XYZ_decoupled_valXYZ(self, verbose=False):
        res = self._test_pattern_trainer_XYZ_valXYZ(procedure="decoupled", verbose=verbose)
        assert(not np.isnan(res[0]))
        assert(np.isnan(res[1])) # because MSE
        assert(not np.isnan(res[2]))
        assert(np.isnan(res[3])) # because MSE

    def test_pattern_trainer_XYZ_prefine_valXYZ(self, verbose=False):
        res = self._test_pattern_trainer_XYZ_valXYZ(procedure="pretrain_finetune", verbose=verbose)
        assert(not np.isnan(res[0]))
        assert(np.isnan(res[1])) # because MSE
        assert(not np.isnan(res[2]))
        assert(np.isnan(res[3])) # because MSE

    #---
    def test_pattern_trainer_XZ_XY_simul_valXZ_XY(self, verbose=False):
        res = self._test_pattern_trainer_XZ_XY_valXYZ(procedure="simultaneous", verbose=verbose)
        assert(not np.isnan(res[0]))
        assert(np.isnan(res[1])) # because MSE
        assert(not np.isnan(res[2]))
        assert(np.isnan(res[3])) # because MSE

    def test_pattern_trainer_XZ_XY_decoupled_valXZ_XY(self, verbose=False):
        res = self._test_pattern_trainer_XZ_XY_valXYZ(procedure="decoupled", verbose=verbose)
        assert(not np.isnan(res[0]))
        assert(np.isnan(res[1])) # because MSE
        assert(not np.isnan(res[2]))
        assert(np.isnan(res[3])) # because MSE

    def test_pattern_trainer_XZ_XY_prefine_valXZ_XY(self, verbose=False):
        res = self._test_pattern_trainer_XZ_XY_valXYZ(procedure="pretrain_finetune", verbose=verbose)
        assert(not np.isnan(res[0]))
        assert(np.isnan(res[1])) # because MSE
        assert(not np.isnan(res[2]))
        assert(np.isnan(res[3])) # because MSE
        
    def test_pattern_trainer_predict(self):
        self._test_pattern_trainer_predict()

# ------------------------------------------------------------------------------

class TestMultiTaskClassificationPattern(TestSinglePatternBase):
    """
      Test the multi task pattern with linear models, but this time with 
      a cross-entropy loss for classification (the other models are regression) 
    """

    def setup(self):
        # define embedded C
        self.C = np.array( [0,1,2,3,4], dtype='int32' ).T

        self.C_val = np.array( [0,1,2,3,], dtype='int32' ).T
        
        # dim of side info
        self.m = None #self.C.shape[1]
        
        # size of desired intermediate representation
        self.d = 1

        super(TestMultiTaskClassificationPattern, self).setup()

    def init_variables(self):
        self.input_var = T.matrix('inputs')
        # do classification
        self.target_var = T.ivector('targets')
        self.side_var = T.ivector('contexts')
        self.num_classes = 2 # number of classes matters
        self.num_side_classes = 5 # number of classes matters
        
        self.side_loss = lasagne.objectives.categorical_crossentropy

    def build_target_loss(self):
        self.target_loss = lasagne.objectives.categorical_crossentropy

    def build_pattern(self):
        self.input_layer = lasagne.layers.InputLayer(shape=(None, self.n),
                                            input_var=self.input_var)
        self.phi = build_linear_simple( self.input_layer, self.d, name="phi")
        self.psi = build_linear_simple( self.phi, self.num_classes, 
            nonlinearity=lasagne.nonlinearities.softmax, name="psi")
        self.beta = build_linear_simple( self.phi, self.num_side_classes, 
            nonlinearity=lasagne.nonlinearities.softmax, name="beta")
            
        self.build_target_loss()
        
        self.pattern = concarne.patterns.MultiTaskPattern(phi=self.phi, psi=self.psi, 
                                             beta=self.beta,
                                             target_var=self.target_var, 
                                             side_var=self.side_var,
                                             target_loss=self.target_loss,
                                             side_loss=self.side_loss
                                             )

    def test_pattern_output(self):
        pass
#         print (self.phi.W.get_value(borrow=True))
#         assert (self.phi.W.get_value(borrow=True).shape == (self.n,self.d))
#         self.phi.W.set_value ( np.array([[1,0]]).T )
#         assert (self.phi.W.get_value(borrow=True).shape == (self.n,self.d))
# 
#         print (self.psi.W.get_value(borrow=True).shape)
#         assert (self.psi.W.get_value(borrow=True).shape == (self.d,self.num_classes))
#         self.psi.W.set_value ( np.array([[1,]]) )
#         assert (self.psi.W.get_value(borrow=True).shape == (self.d,self.num_classes))
# 
#         assert (self.beta.W.get_value(borrow=True).shape == (self.d, self.m))
#         # [1,1] means that we will project the intermediate representation
#         # onto both dimensions of the output representation
#         self.beta.W.set_value ( np.array([[1,1]]) )
#         assert (self.beta.W.get_value(borrow=True).shape == (self.d, self.m))
#         
#         test_prediction = lasagne.layers.get_output(self.pattern, deterministic=True)
#         test_fn = theano.function([self.input_var], test_prediction)
#         X_hat = test_fn(self.X)
#         assert ( np.all(X_hat == self.S) )
# 
#         beta_prediction = lasagne.layers.get_output(self.beta, deterministic=True)
#         beta_fn = theano.function([self.input_var], beta_prediction)
#         C_hat = beta_fn(self.X)
#         assert ( np.all(C_hat[:,0] == self.S[:,0]) )
#         assert ( np.all(C_hat[:,1] == self.S[:,0]) )

    def test_pattern_training_loss_and_grads(self):
        self._test_pattern_training_loss_and_grads()

    def test_learn(self):
        self._test_learn()

    #---
    def test_pattern_trainer_XYZ_simul_no_val(self, verbose=False):
        res = self._test_pattern_trainer_XYZ_no_val(procedure="simultaneous", verbose=verbose)
        assert(not np.isnan(res[0]))
        assert(not np.isnan(res[1])) # because crossentropy!

    def test_pattern_trainer_XYZ_decoupled_no_val(self, verbose=False):
        res = self._test_pattern_trainer_XYZ_no_val(procedure="decoupled", verbose=verbose)
        assert(not np.isnan(res[0]))
        assert(not np.isnan(res[1])) # because crossentropy!

    def test_pattern_trainer_XYZ_prefine_no_val(self, verbose=False):
        res = self._test_pattern_trainer_XYZ_no_val(procedure="pretrain_finetune", verbose=verbose)
        assert(not np.isnan(res[0]))
        assert(not np.isnan(res[1])) # because crossentropy!

    #---
    def test_pattern_trainer_XYZ_simul_valXY(self, verbose=False):
        res = self._test_pattern_trainer_XYZ_valXY(procedure="simultaneous", verbose=verbose)
        assert(not np.isnan(res[0]))
        assert(not np.isnan(res[1])) # because crossentropy!

    def test_pattern_trainer_XYZ_decoupled_valXY(self, verbose=False):
        res = self._test_pattern_trainer_XYZ_valXY(procedure="decoupled", verbose=verbose)
        assert(not np.isnan(res[0]))
        assert(not np.isnan(res[1])) # because crossentropy!

    def test_pattern_trainer_XYZ_prefine_valXY(self, verbose=False):
        res = self._test_pattern_trainer_XYZ_valXY(procedure="pretrain_finetune", verbose=verbose)
        assert(not np.isnan(res[0]))
        assert(not np.isnan(res[1])) # because crossentropy!

    #---
    def test_pattern_trainer_XYZ_simul_valXYZ(self, verbose=False):
        res = self._test_pattern_trainer_XYZ_valXYZ(procedure="simultaneous", verbose=verbose)
        assert(not np.isnan(res[0]))
        assert(not np.isnan(res[1])) # because crossentropy!
        assert(not np.isnan(res[2]))
        assert(not np.isnan(res[3])) # because crossentropy!

    def test_pattern_trainer_XYZ_decoupled_valXYZ(self, verbose=False):
        res = self._test_pattern_trainer_XYZ_valXYZ(procedure="decoupled", verbose=verbose)
        assert(not np.isnan(res[0]))
        assert(not np.isnan(res[1])) # because crossentropy!
        assert(not np.isnan(res[2]))
        assert(not np.isnan(res[3])) # because crossentropy!

    def test_pattern_trainer_XYZ_prefine_valXYZ(self, verbose=False):
        res = self._test_pattern_trainer_XYZ_valXYZ(procedure="pretrain_finetune", verbose=verbose)
        assert(not np.isnan(res[0]))
        assert(not np.isnan(res[1])) # because crossentropy!
        assert(not np.isnan(res[2]))
        assert(not np.isnan(res[3])) # because crossentropy!

    #---
    def test_pattern_trainer_XZ_XY_simul_valXZ_XY(self, verbose=False):
        res = self._test_pattern_trainer_XZ_XY_valXYZ(procedure="simultaneous", verbose=verbose)
        assert(not np.isnan(res[0]))
        assert(not np.isnan(res[1])) # because crossentropy!
        assert(not np.isnan(res[2]))
        assert(not np.isnan(res[3])) # because crossentropy!

    def test_pattern_trainer_XZ_XY_decoupled_valXZ_XY(self, verbose=False):
        res = self._test_pattern_trainer_XZ_XY_valXYZ(procedure="decoupled", verbose=verbose)
        assert(not np.isnan(res[0]))
        assert(not np.isnan(res[1])) # because crossentropy!
        assert(not np.isnan(res[2]))
        assert(not np.isnan(res[3])) # because crossentropy!

    def test_pattern_trainer_XZ_XY_prefine_valXZ_XY(self, verbose=False):
        res = self._test_pattern_trainer_XZ_XY_valXYZ(procedure="pretrain_finetune", verbose=verbose)
        assert(not np.isnan(res[0]))
        assert(not np.isnan(res[1])) # because crossentropy!
        assert(not np.isnan(res[2]))
        assert(not np.isnan(res[3])) # because crossentropy!

    def test_pattern_trainer_predict(self):
        self._test_pattern_trainer_predict()

    def test_pattern_trainer_predict_proba(self):
        self._test_pattern_trainer_predict_proba()
        
# ------------------------------------------------------------------------------


class TestPWTransformationPattern(TestPatternBase):
    """
      Test the pairwise transformation pattern with linear models
    """

    def setup(self):
        # define embedded C
        self.CX = np.array( [[1,2,3,4,5], [2,2,2,2,2]] ).T
        self.Cy = -np.array( [[1,1,1,1,1]] ).T
        
        self.CX_val = np.array( [[1,2,3,4], [2,2,2,2]] ).T
        self.Cy_val = -np.array( [[1,1,1,1]] ).T
        
        # dim of side info input == dim of X
        self.m = self.CX.shape[1]
        
        # size of desired intermediate representation
        self.d = 1

        super(TestPWTransformationPattern, self).setup()
        
        assert(self.CX.shape[1] == self.X.shape[1])
        
    def build_pattern(self):
        self.side_transform_var = T.matrix('side_transforms')
        
        self.input_layer = lasagne.layers.InputLayer(shape=(None, self.n),
                                            input_var=self.input_var)
        self.phi = build_linear_simple( self.input_layer, self.d, name="phi")
        self.psi = build_linear_simple( self.phi, self.num_classes, 
            nonlinearity=None, name="psi")
        self.beta = None
            
        self.build_target_loss()

        self.pattern = concarne.patterns.PairwisePredictTransformationPattern(phi=self.phi, psi=self.psi, 
                                             beta=self.beta,
                                             target_var=self.target_var, 
                                             side_var=self.side_var,
                                             target_loss=self.target_loss,
                                             side_transform_var=self.side_transform_var,
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


    def _test_pattern_trainer_XYZ_no_val(self, procedure='simultaneous',verbose=False):
        self.build_and_run_pattern_trainer(procedure)
        self.trainer.fit_XYZ(self.X, self.Y, [self.CX, self.Cy], verbose=verbose)
        
        return self.trainer.score(self.X, self.Y)
        
    def _test_pattern_trainer_XYZ_valXY(self, procedure='simultaneous', verbose=False):
        self.build_and_run_pattern_trainer(procedure)
        self.trainer.fit_XYZ(self.X, self.Y, [self.CX, self.Cy], X_val=self.X_val, y_val=self.Y_val,
            verbose=verbose)
        
        return self.trainer.score(self.X_val, self.Y_val)

    def _test_pattern_trainer_XYZ_valXYZ(self, procedure='simultaneous', verbose=False):
        self.build_and_run_pattern_trainer(procedure)
        self.trainer.fit_XYZ(self.X, self.Y, [self.CX, self.Cy], X_val=self.X_val, y_val=self.Y_val,
            side_val=[self.X_val, self.CX_val, self.Cy_val], verbose=verbose)
        
        res = []
        res += self.trainer.score(self.X_val, self.Y_val)
        res += self.trainer.score_side([self.X_val, self.CX_val, self.Cy_val])
        return res
         
    def _test_pattern_trainer_XZ_XY_valXYZ (self, procedure='simultaneous', verbose=False):
        self.build_and_run_pattern_trainer(procedure)
        # different length
        XZ = [self.X[1:], self.CX[1:],  self.Cy[1:]]
        XY = [self.X[:-1], self.Y[1:]]
        
        self.trainer.fit_XZ_XY(XZ[0], XZ[1:], XY[0], XY[1],
            X_val=self.X_val, y_val=self.Y_val,
            side_val=[self.X_val, self.CX_val, self.Cy_val], verbose=verbose)
        
        res = []
        res += self.trainer.score(self.X_val, self.Y_val)
        res += self.trainer.score_side([self.X_val, self.CX_val, self.Cy_val])
        
        return res                

    #---
    def test_pattern_trainer_XYZ_simul_no_val(self, verbose=False):
        res = self._test_pattern_trainer_XYZ_no_val(procedure="simultaneous", verbose=verbose)
        assert(not np.isnan(res[0]))
        assert(np.isnan(res[1])) # because MSE

    def test_pattern_trainer_XYZ_decoupled_no_val(self, verbose=False):
        res = self._test_pattern_trainer_XYZ_no_val(procedure="decoupled", verbose=verbose)
        assert(not np.isnan(res[0]))
        assert(np.isnan(res[1])) # because MSE

    def test_pattern_trainer_XYZ_prefine_no_val(self, verbose=False):
        res = self._test_pattern_trainer_XYZ_no_val(procedure="pretrain_finetune", verbose=verbose)
        assert(not np.isnan(res[0]))
        assert(np.isnan(res[1])) # because MSE

    #---
    def test_pattern_trainer_XYZ_simul_valXY(self, verbose=False):
        res = self._test_pattern_trainer_XYZ_valXY(procedure="simultaneous", verbose=verbose)
        assert(not np.isnan(res[0]))
        assert(np.isnan(res[1])) # because MSE

    def test_pattern_trainer_XYZ_decoupled_valXY(self, verbose=False):
        res = self._test_pattern_trainer_XYZ_valXY(procedure="decoupled", verbose=verbose)
        assert(not np.isnan(res[0]))
        assert(np.isnan(res[1])) # because MSE

    def test_pattern_trainer_XYZ_prefine_valXY(self, verbose=False):
        res = self._test_pattern_trainer_XYZ_valXY(procedure="pretrain_finetune", verbose=verbose)
        assert(not np.isnan(res[0]))
        assert(np.isnan(res[1])) # because MSE

    #---
    def test_pattern_trainer_XYZ_simul_valXYZ(self, verbose=False):
        res = self._test_pattern_trainer_XYZ_valXYZ(procedure="simultaneous", verbose=verbose)
        assert(not np.isnan(res[0]))
        assert(np.isnan(res[1])) # because MSE
        assert(not np.isnan(res[2]))
        assert(np.isnan(res[3])) # because MSE

    def test_pattern_trainer_XYZ_decoupled_valXYZ(self, verbose=False):
        res = self._test_pattern_trainer_XYZ_valXYZ(procedure="decoupled", verbose=verbose)
        assert(not np.isnan(res[0]))
        assert(np.isnan(res[1])) # because MSE
        assert(not np.isnan(res[2]))
        assert(np.isnan(res[3])) # because MSE

    def test_pattern_trainer_XYZ_prefine_valXYZ(self, verbose=False):
        res = self._test_pattern_trainer_XYZ_valXYZ(procedure="pretrain_finetune", verbose=verbose)
        assert(not np.isnan(res[0]))
        assert(np.isnan(res[1])) # because MSE
        assert(not np.isnan(res[2]))
        assert(np.isnan(res[3])) # because MSE

    #---
    def test_pattern_trainer_XZ_XY_simul_valXZ_XY(self, verbose=False):
        res = self._test_pattern_trainer_XZ_XY_valXYZ(procedure="simultaneous", verbose=verbose)
        assert(not np.isnan(res[0]))
        assert(np.isnan(res[1])) # because MSE
        assert(not np.isnan(res[2]))
        assert(np.isnan(res[3])) # because MSE

    def test_pattern_trainer_XZ_XY_decoupled_valXZ_XY(self, verbose=False):
        res = self._test_pattern_trainer_XZ_XY_valXYZ(procedure="decoupled", verbose=verbose)
        assert(not np.isnan(res[0]))
        assert(np.isnan(res[1])) # because MSE
        assert(not np.isnan(res[2]))
        assert(np.isnan(res[3])) # because MSE

    def test_pattern_trainer_XZ_XY_prefine_valXZ_XY(self, verbose=False):
        res = self._test_pattern_trainer_XZ_XY_valXYZ(procedure="pretrain_finetune", verbose=verbose)
        assert(not np.isnan(res[0]))
        assert(np.isnan(res[1])) # because MSE
        assert(not np.isnan(res[2]))
        assert(np.isnan(res[3])) # because MSE                
        
    def test_pattern_trainer_predict(self):
        self._test_pattern_trainer_predict()
        
# ------------------------------------------------------------------------------
if __name__ == "__main__":
    td = TestDirectPattern()
    #td = TestMultiTaskPattern()
    #td = TestPWTransformationPattern()
#     td = TestMultiTaskClassificationPattern()
    td.setup()
#    td.test_pattern_output()
#    td.test_pattern_training_loss_and_grads()
#    td.test_learn()
    
    td.test_pattern_trainer_predict()

    verbose=False
        

#    td.test_pattern_trainer_XYZ_simul_no_val(verbose=verbose)
#    td.test_pattern_trainer_XYZ_decoupled_no_val(verbose=verbose)
#    td.test_pattern_trainer_XYZ_prefine_no_val(verbose=verbose)
#
#    td.test_pattern_trainer_XYZ_simul_valXY(verbose=verbose)
#    td.test_pattern_trainer_XYZ_decoupled_valXY(verbose=verbose)
#    td.test_pattern_trainer_XYZ_prefine_valXY(verbose=verbose)
#    
#    td.test_pattern_trainer_XYZ_simul_valXYZ(verbose=verbose)
#    td.test_pattern_trainer_XYZ_decoupled_valXYZ(verbose=verbose)
#    td.test_pattern_trainer_XYZ_prefine_valXYZ(verbose=verbose)    
#
#    td.test_pattern_trainer_XZ_XY_simul_valXZ_XY(verbose=verbose)
#    td.test_pattern_trainer_XZ_XY_decoupled_valXZ_XY(verbose=verbose)
#    td.test_pattern_trainer_XZ_XY_prefine_valXZ_XY(verbose=verbose)
