# -*- coding: utf-8 -*-

from __future__ import print_function
from __future__ import absolute_import

from .iterators import AlignedBatchIterator
from .utils import all_elements_equal_len, isiterable, generate_timestamp

import lasagne

import concarne.lasagne
from concarne.lasagne import score_categorical_crossentropy,\
    score_multivariate_categorical_crossentropy
from concarne.utils import isfunction

import theano
import theano.tensor as T

import numpy as np

import time
import copy

from warnings import warn

class PatternTrainer(object):
    """The :class:`PatternTrainer` provides a simple way of training any given pattern 
       consisting of lasagne layers as functions using mini batch stochastic 
       gradient descent (SGD)
       
       It is similar to :class:`lasagne.layers.Layer` and mimics some of 
       its functionality, but does not inherit from it.
       
       Example with aligned data X_train, y_train and Z_train (e.g. multitask)::
       
        > pt = concarne.training.PatternTrainer(pattern, 
                     num_epochs=3, batch_size=50, 
                     update_learning_rate=0.0001,
                     target_weight=0.9, side_weight=0.1)
        > pt.fit_XYZ(X_train, y_train, [Z_train], X_val=X_val, y_val=y_val,
                side_val=[X_val, Z_val], verbose=True)
            Training procedure: simultaneous
             Optimize phi & psi & beta using a weighted sum of target and side objective
               -> standard mode with single training function
            Starting training...
            Optimize phi & psi & beta using a weighted sum of target and side objective
             Update: nesterov_momentum(learning_rate=0.0001, momentum=0.9) 
               -> standard mode with single training function
            Starting training...
            Epoch 1 of 3 took 0.010s
              training loss:                15.220120
               (target: 10.156144,  side: 60.795899 (absolute))
              validation loss:              6.142665
              validation accuracy:          52.20 %
              side validation loss:         348.806490
              side validation accuracy:     21.40 %
            Epoch 2 of 3 took 0.008s
              training loss:                10.727928
               (target: 4.670905,  side: 65.241137 (absolute))
              validation loss:              6.281717
              validation accuracy:          47.80 %
              side validation loss:         331.070974
              side validation accuracy:     25.10 %
            Epoch 3 of 3 took 0.008s
              training loss:                9.908304
               (target: 3.530447,  side: 67.309020 (absolute))
              validation loss:              6.379712
              validation accuracy:          43.80 %
              side validation loss:         313.270051
              side validation accuracy:     24.80 %
            Storing pattern to pt_2016-04-27_22-01-22-142855_simultaneous

       You can then easily compute the target and side loss for test data::

        > loss, acc = pt.score(X_test, y_test, verbose=True)
            Score:
              loss:                 4.284542
              accuracy:             31.21 %            
          side_loss, side_acc = pt.score_side([X_test, Z_test], verbose=True)
             Score for side loss:
              loss:                 285.109184
              accuracy:             16.21 %            

       If in your task, you have more (or simply a different amount of)
       side information available than labels, you can use the method 
       fit_XZ_XY::
       
        > pt.fit_XZ_XY(X_train_unlabeled, [Z_train_for_unlabeled], 
                          X_train_labeled, y_train_for_labeled, 
                          X_val=X_val, y_val=y_val)
        
       In the simultaneous procedure, instead of jointly optimizing the
       gradient for combined objective, we alternate the computation of the
       gradient for the side and the target objectives (minibatch-wise).

       Parameters
       ----------       
        pattern :  :class:`Pattern`
            Instance of a pattern
        procedure: {'decoupled', 'pretrain_finetune', 'simultaneous'}
            Three training procedures are supported. decoupled and 
            pretrain_finetune both use a two stage process, whereas
            simultaneous optimizes a linear combination of the target and
            side losses. decoupled is only applicable if the
            side loss provides enough guidance to learn a good 
            representation s, since in the second training phase 
            :math:`\phi` is not changed anymore. In pretrain_finetune,
            :math:`\phi` is also optimized in the second training phase.
        update: lasagne.updates.***
            Update method for parameters after epoch. Default: nesterov_momentum.
            Additional params for the update method should be provided using
            keyword arguments 'update_<arg_name>'.
        XZ_update: lasagne.updates.***
            Different update method for parameters in the decoupled or pretrain_finetune
            procedures (ignored in simultaneous), optional.
            Additional params for the update method should be provided using
            keyword arguments 'XZ_update_<arg_name>'.            
        batch_size: int
            Batch size for SGD.
            Irrelevant if you specify the iterator for the fit_*** methods yourself.
        XZ_batch_size: int
            Different batch size for SGD in the decoupled or pretrain_finetune
            procedures (ignored in simultaneous).
            Irrelevant if you specify the iterator for the fit_*** methods yourself.
        num_epochs :  int
            Number of epochs used to run SGD
        XZ_num_epochs : int
            Number of epochs to run for XZ in the decoupled or pretrain_finetune
            procedures (ignored in simultaneous), optional
        XYpsi_num_epochs : int
            Number of epochs to run for XY optimizing only psi (keeping phi fixed)
            in the pretrain_finetune. It can be thought of an intermediate 
            third training stage after pre-training with the side information,
            but before training both psi and phi with the targets.
            It can be sometimes helpful because psi is usually initialized randomly,
            and will therefore produce mostly wrong predicitions, leading to large errors;
            propagating these errors through phi might destroy the pre-training.
            Default value is 0.
        target_weight: float, optional
            only required for simultaneous training procedure
        side_weight: float, optional
            only required for simultaneous training procedure
        verbose: bool, deprecated
            Use the verbose flags in fit_*** and score instead
        save_params: bool
            If true the parameters of the function will be stored.
            This is particularly interesting for the decoupled and pre-train-fine-tune
            procedures, as it allows to inspect intermediate results.
            The file format is 'pt_<timestamp>.tar'.
        """
    def __init__(self, pattern, 
                 procedure="simultaneous", 
                 update=lasagne.updates.nesterov_momentum,
                 XZ_update=None,
                 batch_size=100,
                 XZ_batch_size=None,
                 num_epochs=500, 
                 XZ_num_epochs=None,
                 XYpsi_num_epochs=None,
                 target_weight=None, 
                 side_weight=None,
                 verbose=None,
                 save_params=False,
                 **kwargs):
        self.pattern = pattern
        self.procedure = procedure
        assert (procedure in ["simultaneous", "decoupled", "pretrain_finetune"])

        self.update = update
        
        self.XZ_update = self.update
        if XZ_update is not None:
            self.XZ_update = XZ_update
            if procedure == "simultaneous":
                warn("XZ_update has been provided, but procedure is 'simultaneous'; "
                     "ignoring")
        
        self.num_epochs = num_epochs

        self.XZ_num_epochs = num_epochs
        if XZ_num_epochs is not None:
            self.XZ_num_epochs = XZ_num_epochs
            if procedure == "simultaneous":
                warn("XZ_num_epochs has been provided, but procedure is 'simultaneous'; "
                     "ignoring")

        self.batch_size = batch_size

        self.XZ_batch_size = batch_size
        if XZ_batch_size is not None:
            self.XZ_batch_size = XZ_batch_size
            if procedure == "simultaneous":
                warn("XZ_batch_size has been provided, but procedure is 'simultaneous'; "
                     "ignoring")

        self.XYpsi_num_epochs = 0
        if XYpsi_num_epochs is not None:
            self.XYpsi_num_epochs = XYpsi_num_epochs
            if procedure != "pretrain_finetune":
                warn("XYpsi_num_epochs has been provided, but procedure is NOT "
                     "'pretrain_finetune'; ignoring")
        
        # target_weight and side_weight are only relevant for simultaneous
        self.loss_weights = {}
        if self.procedure == "simultaneous":
            if target_weight is not None:
                self.loss_weights['target_weight'] = target_weight
            if side_weight is not None:
                self.loss_weights['side_weight'] = side_weight
        
        self.test_objective_fn = pattern.target_loss_fn
        if self.test_objective_fn is None and pattern.target_loss_fn is not None:
            warn("You initialized your pattern with a custom target_loss, "
                +"i.e. not a python function object, but probably a theano "
                +"expression. PatternTrainer will not be able to compute "
                +"the validation/test accuracy")
            self.test_objective_fn = pattern.target_loss
        elif self.test_objective_fn is None:
            raise Exception("Unable to infer target objective.")
                
        self.side_test_objective_fn = pattern.side_loss_fn
        if self.side_test_objective_fn is None and pattern.side_loss_fn is not None:
            warn("You initialized your pattern with a custom side_loss, "
                +"i.e. not a python function object, but probably a theano "
                +"expression. PatternTrainer will not be able to compute "
                +"the validation/test accuracy")
            self.side_test_objective_fn = pattern.side_loss
        elif self.side_test_objective_fn is None:
            raise Exception("Unable to infer side objective.")
        
        self.val_fn = None
        self.val_batch_iterator = None

        self.side_val_fn = None
        self.side_val_batch_iterator = None
        self._side_val_fn_not_supported = False
                        
        self.save_params = save_params
        if self.save_params:
            ts = generate_timestamp()
            self._dump_filename = "pt_%s" % ts

        # deprecation warnings
        if verbose is not None:
            warn("passing verbose to constructor of PatternTrainer is deprecated."+
                " Use verbose flags for fit*** and score methods instead." )

        if 'learning_rate' in kwargs:
            warn("The 'learning_rate' argument has been deprecated, please use "
                 "the 'update_learning_rate' parameter instead and make sure you use "
                 "'update=nesterov_momentum'" )
            self.update_learning_rate = kwargs['learning_rate']
            del kwargs['learning_rate']

        if 'momentum' in kwargs:
            warn("The 'momentum' argument has been deprecated, please use "
                 "the 'update_momentum' parameter instead and make sure you use "
                 "'update=nesterov_momentum'" )
            self.update_momentum = kwargs['momentum']
            del kwargs['momentum']

        # store remaining kw_args
        for key in kwargs.keys():
            assert not hasattr(self, key)
        vars(self).update(kwargs)
        self._kwarg_keys = list(kwargs.keys())


    def _get_params_for(self, name):
        """This method has been adapted from the NeuralFit class in nolearn.
        https://github.com/dnouri/nolearn/blob/master/nolearn/lasagne/base.py
        Copyright (c) 2012-2015 Daniel Nouri"""
        
        collected = {}
        prefix = '{}_'.format(name)

        params = vars(self)
        #more_params = self.more_params

        #for key, value in itertools.chain(params.items(), more_params.items()):
        for key, value in params.items():
            if key.startswith(prefix):
                collected[key[len(prefix):]] = value

        return collected

    def _compile_train_fn(self, train_fn_inputs, loss_weights, tags, update, **update_params):
        loss, target_loss, side_loss = self.pattern.training_loss(all_losses=True, **loss_weights)
        loss = loss.mean()
        
        # collect output dictionary
        outputs = {}
        outputs['loss'] = loss
        try:
            outputs['target_loss'] =  target_loss.mean()
        except:
            pass
            
        try:
            outputs['side_loss'] =  side_loss.mean()
        except:
            pass
        
        # -----
        # get trainable params and build update
        params = lasagne.layers.get_all_params(self.pattern, trainable=True, **tags)
        updates = update(loss, params, **update_params)
    
#        print (params)
#        print (updates)
    
        # Compile a function performing a training step on a mini-batch (by giving
        # the updates dictionary) and returning the corresponding training loss:
        
        # we switch of the on_unused_input warning because in some
        # circumstances, e.g. in the decoupled procedure, we might pass
        # unused inputs to the training function (e.g. target_var
        # is not used in the pretrain phase). The only reason for having this
        # so we don't have to deal with implementing different 
        # training_input_vars methods in the Pattern class
        theano.config.on_unused_input = 'ignore'

        # finally: build theano training function
        train_fn = theano.function(train_fn_inputs, outputs, updates=updates)
        
        # FIXME
        #self._train_fn_no_update = theano.function(train_fn_inputs, outputs)
    
        return train_fn

    def _compile_val_fn(self):
        return self.__compile_val_fn([self.pattern.input_var], self.pattern.target_var,
            lasagne.layers.get_output(self.pattern, deterministic=True),
            self.test_objective_fn)

    def _compile_side_val_fn(self):
        side_input_vars = self.pattern.side_input_vars
        side_target_var = self.pattern.side_target_var

        if side_target_var is not None:
            return self.__compile_val_fn(side_input_vars, side_target_var,
                self.pattern.get_beta_output_for(deterministic=True),
                self.side_test_objective_fn)

        else:
            _, _, side_loss = self.pattern.training_loss(all_losses=True)
            return theano.function(side_input_vars, [side_loss.mean()])

        
    def __compile_val_fn(self, input_vars, target_var, 
            test_prediction,
            test_objective):
            
        if not isiterable(input_vars):
            input_vars = [input_vars]
        if type(input_vars) == tuple:
            input_vars = list(input_vars)
        
        # special case: if we use the squared_error loss, but target_var is a vector
        # (1 dim target) we flatten the prediction -- otherwise we get a theano error
        if test_objective == lasagne.objectives.squared_error and \
            target_var.type == T.dvector or target_var.type == T.dvector:
            test_prediction = test_prediction.flatten()
        
        test_acc = None
        if isfunction(test_objective):
            test_loss = test_objective(test_prediction, target_var).mean()
            
            # Create an expression for the classification accuracy
            if test_objective == lasagne.objectives.categorical_crossentropy:
                test_acc = score_categorical_crossentropy(test_prediction, target_var)
            elif test_objective == concarne.lasagne.multivariate_categorical_crossentropy:
                test_acc = score_multivariate_categorical_crossentropy(test_prediction, target_var)
        else:
            test_loss = test_objective
    
        outputs = [test_loss]
        if test_acc is not None:
            outputs.append(test_acc)
    
        # FIXME
        #outputs.append(test_objective(test_prediction, target_var))
    
        # Compile a second function computing the validation loss and accuracy:
        inputs = input_vars + [target_var]
        
        val_fn = theano.function(inputs, outputs)

#         print (inputs)
#         print (outputs)

        return val_fn
        
    def fit_XYZ(self, X, Y, Zs,
            batch_iterator=None, 
            X_val=None, y_val=None,
            side_val=None,
            verbose=False):
        """Training function for aligned data (same number of examples for
       X, Y and Z)

       Parameters
       ----------       
        X :  numpy array
            Input data (rows: samples, cols: features)
        Y :  numpy array
            Labels / target data
        Zs:  list of numpy arrays/lists
            Side information - even if the pattern / iterator only expects the
            value for one side variable, you MUST give a list here
        batch_iterator: iterator, optional
            Your custom iterator class that accepts X,Y,Z1,..Zn as inputs
        X_val: numpy array, optional
            Validation input data
        y_val: numpy array, optional
            Validation labeled data
        side_val: list of numpy arrays, optional
            Validation data for side information 
        verbose: bool / int
            Verbosity level (default 0/False)
        """

#         if not isiterable(Zs):
        if type(Zs) != list and type(Zs) != tuple:
            raise Exception("Make sure that you provide a list of side "
                + " information Zs, even if the pattern only expects one side variable")

        batch_iterators = []
        if batch_iterator is None:
            batch_iterator = AlignedBatchIterator(self.batch_size, shuffle=True)
            batch_iterators.append(batch_iterator)

        if self.batch_size != self.XZ_batch_size:
            batch_iterator_XZ = AlignedBatchIterator(self.XZ_batch_size, shuffle=True)
            batch_iterators.append(batch_iterator_XZ)
        else:
            # twice the same
            batch_iterators.append(batch_iterator)

        batch_iterator_args = [X, Y] + Zs
        if not all_elements_equal_len(batch_iterator_args):
            raise Exception("X, Y and Z must have same len!")

        return self._fit(batch_iterators, [batch_iterator_args]*2, 
                         "XYZ", X_val, y_val, side_val, verbose)


        
    def fit_XZ_XY(self, X1, Zs, X2, Y,
            batch_iterator_XZ=None,
            batch_iterator_XY=None, 
            X_val=None, y_val=None,
            side_val=None,
            verbose=False):
        """Training function for unaligned data (one data set for X and Z,
           another data set for X and Y)

       Parameters
       ----------       
        X1 :  numpy array
            Input data use for optimizing side objective 
            (rows: samples, cols: features)
        Zs:  list of numpy arrays/lists
            Side information - even if the pattern / iterator only expects the
            value for one side variable, you MUST give a list here
        X1 :  numpy array
            Input data use for optimizing target objective 
            (rows: samples, cols: features)
        Y :  numpy array
            Labels / target data (per default each array should have same len as X1)
        batch_iterator_XZ: iterator, optional
            Your custom iterator for going through the X-Z data
        batch_iterator_XY: iterator, optional
            Your custom iterator for going through the X-Y data
        X_val: numpy array, optional
            Validation input data
        y_val: numpy array, optional
            Validation labeled data
        side_val: list of numpy arrays, optional
            Validation data for side information 
            (useful in predictive patterns, e.g. multi-task)
        verbose: bool / int
            Verbosity level (default 0/False)
        """
        
        #if not isiterable(Zs):
        if type(Zs) != list and type(Zs) != tuple:
            raise Exception("Make sure that you provide a list of side "
                + " information Zs, even if the pattern only expects one side variable")

        if batch_iterator_XY is None:
            batch_iterator_XY = AlignedBatchIterator(self.batch_size, shuffle=True)
        if batch_iterator_XZ is None:
            batch_iterator_XZ = AlignedBatchIterator(self.XZ_batch_size, shuffle=True)

        batch_iterators = [
            batch_iterator_XZ,
            batch_iterator_XY
        ]

        batch_iterator_args_lst = [ [X1] + Zs, [X2, Y] ]
        if not all_elements_equal_len(batch_iterator_args_lst[0]):
            raise Exception("X1 and all entries in Zs must have same len!")
        if not all_elements_equal_len(batch_iterator_args_lst[1]):
            raise Exception("X2 and Y must have same len!")
            
        return self._fit(batch_iterators, batch_iterator_args_lst, 
                         "XZ_XY", X_val, y_val, side_val, verbose)

    def __update_info(self, update, update_params):
        return ("Update: %s(%s) " % (update.__name__,
          ", ".join([ "%s=%s" % (k, str(v)) for k,v in update_params.items() ])))

        
    def _fit(self, batch_iterators, batch_iterator_args_lst, data_alignment="XYZ", 
                X_val=None, y_val=None, side_val=None, verbose=False,):

        assert (data_alignment in ["XYZ", "XZ_XY"])

#         if side_val is not None:
#             # FIXME
#             warn("side_val not yet supported; ignoring")
#             side_val = None

        # check length of validation args
        if X_val is not None:
            val_args = [X_val, y_val]
#             if side_val is not None:
#                 val_args += Zs_val
            if not all_elements_equal_len(val_args):
                raise Exception("X_val and Y_val must have same len!")
                            
        #if X_val is not None and y_val is not None:
        #    assert (len(X_val) == len(y_val))

        # training procedures
        if verbose:
            print ("Training procedure: %s" % self.procedure)

        # default: only one phase, with all vars as inputs for train_fn
        train_vars_phase1 = self.pattern.training_input_vars
        train_vars_phase2 = train_vars_phase1
        if data_alignment == "XZ_XY":
            # alternating: two train_fn, one accepting X,Z, two accepting X,Y
            train_vars_phase1 = [self.pattern.input_var] + list(self.pattern.side_vars) # XZ
            train_vars_phase2 = [self.pattern.input_var, self.pattern.target_var] # XY

        update_params = self._get_params_for('update')
        update_params_XZ = self._get_params_for('XZ_update')
        if len(update_params_XZ) == 0: # Assumining u
            update_params_XZ = update_params

        # ========================================================
        if self.procedure in ['decoupled', 'pretrain_finetune']:
            # ---------------------
            # first training phase
            if verbose:
                print ("Optimize phi & beta using the side objective")
                print (" "+self.__update_info(self.XZ_update, update_params_XZ))
                
            train_fn = self._compile_train_fn(train_vars_phase1,
                                              loss_weights={'target_weight': 0.0, 'side_weight': 1.0}, 
                                              tags= {'psi': False}, 
                                              update=self.XZ_update,
                                              **update_params_XZ)
            # passing X_val and y_val doesn't make sense because psi is not trained
            self._train([train_fn], [batch_iterators[0]], [batch_iterator_args_lst[0]], self.XZ_num_epochs, 
                        side_val=side_val, verbose=verbose)
            
            if self.save_params:
                df = self._dump_filename + "_" + self.procedure + "_phase1"
                if verbose:
                    print ("Storing pattern after phase 1 to %s" % df)
                self.pattern.save(df)
            
            # ---------------------
            # second training phase
            if self.procedure == 'decoupled':
                if verbose:
                    print ("=====\nOptimize psi using the target objective")
                    print (" "+self.__update_info(self.update, update_params))
                train_fn = self._compile_train_fn(train_vars_phase2,
                                                  loss_weights={'target_weight': 1.0, 'side_weight': 0.0}, 
                                                  tags= {'phi': False, 'beta': False },  # beta: False is implicit
                                                  update=self.update,
                                                  **update_params)
                self._train([train_fn], [batch_iterators[1]], [batch_iterator_args_lst[1]], self.num_epochs, 
                        X_val=X_val, y_val=y_val, verbose=verbose)
                
            elif self.procedure == 'pretrain_finetune':
                # intermediate training phase - train psi only, keeping phi fixed
                if self.XYpsi_num_epochs > 0:
                    if verbose:
                        print ("=====\nOptimize psi using the target objective")
                        print (" "+self.__update_info(self.XZ_update, update_params_XZ))
                    train_fn = self._compile_train_fn(train_vars_phase2,
                                                      loss_weights={'target_weight': 1.0, 'side_weight': 0.0}, 
                                                      tags= {'psi': True}, 
                                                      update=self.XZ_update,
                                                      **update_params_XZ)
                    self._train([train_fn], [batch_iterators[1]], [batch_iterator_args_lst[1]], self.XYpsi_num_epochs, 
                        X_val=X_val, y_val=y_val, verbose=verbose)

                    if self.save_params:
                        df = self._dump_filename + "_" + self.procedure + "_phase2_psionly"
                        if verbose:
                            print ("Storing pattern after phase 2 (psi only) to %s" % df)
                        self.pattern.save(df)
                    
                if verbose:
                    print ("=====\nOptimize phi & psi using the target objective")
                    print (" "+self.__update_info(self.update, update_params))
                train_fn = self._compile_train_fn(train_vars_phase2,
                                                  loss_weights={'target_weight': 1.0, 'side_weight': 0.0},
                                                  tags= {'beta': False}, 
                                                  update=self.XZ_update,
                                                  **update_params_XZ)
                self._train([train_fn], [batch_iterators[1]], [batch_iterator_args_lst[1]], self.num_epochs, 
                    X_val=X_val, y_val=y_val, side_val=side_val, verbose=verbose)

            if self.save_params:
                df = self._dump_filename + "_" + self.procedure + "_phase2"
                if verbose:
                    print ("Storing pattern after phase 2 to %s" % df)
                self.pattern.save(df)

        # ========================================================
        elif self.procedure == 'simultaneous':
            if verbose:
                print ("Optimize phi & psi & beta using a weighted sum of target and side objective")
                print (" "+self.__update_info(self.update, update_params))
            if data_alignment == "XYZ":
                if verbose:
                    print ("   -> standard mode with single training function")
                train_fn = self._compile_train_fn(self.pattern.training_input_vars,
                                                  loss_weights=self.loss_weights,
                                                  tags= {},
                                                  update=self.update,
                                                  **update_params)
                train_fn = [train_fn]*2
            else:
                if verbose:
                    print ("   -> alternating mode with two training functions")
                lw1 = copy.copy(self.loss_weights)
                lw1['target_weight'] = 0.
                train_fn1 = self._compile_train_fn(train_vars_phase1,
                    loss_weights=lw1, 
                    tags= {'psi': False},
                    update=self.update,
                    **update_params)

                lw2 = copy.copy(self.loss_weights)
                lw2['side_weight'] = 0.
                train_fn2 = self._compile_train_fn(train_vars_phase2,
                    loss_weights=lw2, 
                    tags= {'beta': False},
                    update=self.update,
                    **update_params)

                train_fn = [train_fn1, train_fn2]
                
            self._train(train_fn, batch_iterators, batch_iterator_args_lst, self.num_epochs, 
                    X_val=X_val, y_val=y_val, side_val=side_val, verbose=verbose)

            if self.save_params:
                df = self._dump_filename + "_" + self.procedure
                if verbose:
                    print ("Storing pattern to %s" % df)
                self.pattern.save(df)
        
        return self

    def _train(self, train_fns, batch_iterators, batch_iterator_args_lst, num_epochs, 
                X_val=None, y_val=None, side_val=None, verbose=False):

        if self.val_fn is None:
            self.val_fn = self._compile_val_fn()
        
        if side_val is not None and self.side_val_fn is None and not self._side_val_fn_not_supported:
            try:
                self.side_val_fn = self._compile_side_val_fn()
            except NotImplementedError, e:
                warn("The pattern does not support score_side %s" % e)
                self._side_val_fn_not_supported = True
                side_val = None
        
        if not isiterable(train_fns):
            train_fns = [train_fns]
        if not isiterable(batch_iterators):
            batch_iterators = [batch_iterators]
        if not isiterable(batch_iterator_args_lst):
            batch_iterator_args_lst = [batch_iterator_args_lst]
        
        no_unique_train_fn = len(set(train_fns))
        
        if verbose:
            print("Starting training...")
        # We iterate over epochs:
        for epoch in range(num_epochs):
            # In each epoch, we do a full pass over the training data:
            train_err = 0
            train_target_err = 0
            train_side_err = 0
            train_batches = 0
            start_time = time.time()
            
            for train_fn, batch_iterator, batch_iterator_args \
                in zip(train_fns, batch_iterators, batch_iterator_args_lst):
                for batch in batch_iterator(*batch_iterator_args):
                    outputs = train_fn(*batch)
                    #outputs = self._train_fn_no_update(*batch) # for testing
                    train_err += outputs['loss']
                    train_batches += 1
                    if 'target_loss' in outputs:
                        train_target_err += outputs['target_loss']
                    if 'side_loss' in outputs:
                        train_side_err += outputs['side_loss']
            
            # normalize by number of training functions                        
            train_batches /= no_unique_train_fn


            # And a pass over the validation data:
            if X_val is not None and y_val is not None:
                bs = batch_iterators[-1].batch_size
                val_err, val_acc = self.score(X_val, y_val, batch_size=bs)
                #print (self.score(batch_iterator_args[0], batch_iterator_args[1], batch_size=bs))
    
            if side_val is not None:
                side_val_err, side_val_acc = self.score_side(side_val)
    
            # Then we print the results for this epoch:
            if verbose:
                if len(self.loss_weights) > 0:
                    # normalize wrt weights
                    train_target_err /= self.loss_weights['target_weight']
                    train_side_err /= self.loss_weights['side_weight']
            
                print("Epoch {} of {} took {:.3f}s".format(
                    epoch + 1, num_epochs, time.time() - start_time))
                print("  training loss:\t\t{:.6f}".format(train_err / train_batches))
                print("   (target: {:.6f}, ".format(train_target_err / train_batches)
                     + " side: {:.6f}".format(train_side_err / train_batches) + " (absolute))")
    
                if X_val is not None and y_val is not None:
                    print("  validation loss:\t\t{:.6f}".format(val_err))
                    print("  validation accuracy:\t\t{:.2f} %".format(val_acc * 100))

                if side_val is not None:
                    print("  side validation loss:\t\t{:.6f}".format(side_val_err))
                    print("  side validation accuracy:\t{:.2f} %".format(side_val_acc * 100))
                                        
    def score(self, X, y, batch_size=None, verbose=False):
        """
        Parameters
        ----------       
        X :  numpy array
            Input data (rows: samples, cols: features)
        y :  numpy array
            Target labels
        batch_size: int, optional
            batch size for score
        verbose: string, optional
            whether to print results of score (default: false)
        """
        
        if self.val_fn is None:
            self.val_fn = self._compile_val_fn()
        
        assert (len(X) == len(y))
        
        if batch_size is None:
            batch_size = len(X)
        
        # simple iterator for valid/test error
        if self.val_batch_iterator is None or self.val_batch_iterator.batch_size != batch_size:
            self.val_batch_iterator = AlignedBatchIterator(batch_size, shuffle=False)
        
        # After training, we compute and print the test error:
        test_err = 0
        test_acc = 0
        test_batches = 0
        for batch in self.val_batch_iterator(X, y):
            inputs, targets = batch
            res = self.val_fn(inputs, targets)

            if len(res) == 2:
                err, acc = res
            else:
                assert (len(res) == 1)
                if isiterable(res[0]):
                    assert (len(res) == 1)
                    err = res[0]
                else:
                    err = res
                acc = np.nan

#             if len(res) == 2:
#                 err, acc = res
#             else:
#                 err = res
#                 acc = np.nan
                
            test_err += err
            test_acc += acc
            test_batches += 1
    
        if test_batches > 0 and verbose:
            print("Score:")
            print("  loss:\t\t\t{:.6f}".format(test_err / test_batches))
            # TODO do not print this if val_fn does not return accuracy
            print("  accuracy:\t\t{:.2f} %".format(
                test_acc / test_batches * 100))                

        return test_err/test_batches, test_acc/test_batches

        
    def score_side(self, inputs, batch_size=None, verbose=False):
        """
        Experimental: compute the side score
        
        Parameters
        ----------       
        inputs :  list or tuple of numpy arrays
            All data required to compute the scores.
            Should match the pattern.
        batch_size: int, optional
            batch size for score
        verbose: string, optional
            whether to print results of score (default: false)
        """
        
        try:
            if self.side_val_fn is None:
                self.side_val_fn = self._compile_side_val_fn()
        except NotImplementedError, e:
            warn("The pattern does not support score_side %s" % e)
            return None
        
        if not isiterable(inputs):
            inputs = [inputs]
        
        if batch_size is None:
            batch_size = len(inputs[0])
        
        # simple iterator for valid/test error
        if self.side_val_batch_iterator is None or self.side_val_batch_iterator.batch_size != batch_size:
            self.side_val_batch_iterator = AlignedBatchIterator(batch_size, shuffle=False)
        
        # After training, we compute and print the test error:
        test_err = 0
        test_acc = 0
        test_batches = 0
        for batch in self.side_val_batch_iterator(*inputs):
            res = self.side_val_fn(*batch)

            if len(res) == 2:
                err, acc = res
            else:
                assert (len(res) == 1)
                if isiterable(res[0]):
                    assert (len(res) == 1)
                    err = res[0]
                else:
                    err = res
                acc = np.nan
            test_err += err
            test_acc += acc
            test_batches += 1
    
        if test_batches > 0 and verbose:
            print("Score for side:")
            print("  loss:\t\t\t{:.6f}".format(test_err / test_batches))
            # TODO do not print this if side_val_fn does not return accuracy
            print("  accuracy:\t\t{:.2f} %".format(
                (test_acc / test_batches) * 100))                

        return test_err/test_batches, test_acc/test_batches
        