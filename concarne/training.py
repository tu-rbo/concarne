# -*- coding: utf-8 -*-

from __future__ import print_function

from .iterators import AlignedBatchIterator
from .utils import all_elements_equal_len, isiterable

import lasagne
import theano
import theano.tensor as T

import time
import copy

class PatternTrainer(object):
    
    def __init__(self, pattern, 
                 num_epochs, 
                 learning_rate,
                 batch_size,
                 momentum=0.9,
                 procedure="simultaneous", 
                 target_weight=None, 
                 context_weight=None,
                 test_objective=lasagne.objectives.categorical_crossentropy,
                 verbose=True):
        self.pattern = pattern
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.procedure = procedure
        assert (procedure in ["simultaneous", "decoupled", "pretrain_finetune"])
        
        # target_weight and context_weight are only relevant for simultaneous
        self.loss_weights = {}
        if self.procedure == "simultaneous":
            if target_weight is not None:
                self.loss_weights['target_weight'] = target_weight
            if context_weight is not None:
                self.loss_weights['context_weight'] = context_weight
        
        self.test_objective = test_objective
        
        self.val_fn = None
        self.val_batch_iterator = None
        
        self.verbose = verbose

    def _compile_train_fn(self, train_fn_inputs, loss_weights, tags):
        loss = self.pattern.training_loss(**loss_weights).mean()
        
        params = lasagne.layers.get_all_params(self.pattern, trainable=True, **tags)
        
        # TODO add possibility to use different update mechanism
        updates = lasagne.updates.nesterov_momentum(
            loss, params, learning_rate=self.learning_rate, momentum=self.momentum)
    
        # Compile a function performing a training step on a mini-batch (by giving
        # the updates dictionary) and returning the corresponding training loss:
        train_fn = theano.function(train_fn_inputs, loss, updates=updates)
    
        return train_fn

    def _compile_val_fn(self):
        input_var = self.pattern.input_var
        target_var = self.pattern.target_var
        
        test_prediction = lasagne.layers.get_output(self.pattern, deterministic=True)
        test_loss = self.test_objective(test_prediction, target_var).mean()
        # Create an expression for the classification accuracy
        if self.test_objective == lasagne.objectives.categorical_crossentropy:
            test_acc = T.mean(T.eq(T.argmax(test_prediction, axis=1), target_var),
                              dtype=theano.config.floatX)
        else: 
            test_acc = None
    
        outputs = [test_loss]
        if test_acc is not None:
            outputs.append(test_acc)
    
        # Compile a second function computing the validation loss and accuracy:
        val_fn = theano.function([input_var, target_var], outputs)

        return val_fn
        
    def fit_XYC(self, X, Y, Cs,
            batch_iterator=None, 
            X_val=None, y_val=None):

        if not isiterable(Cs) or len(Cs[0]) != len(X):
            Cs = [Cs]

        if batch_iterator is None:
            batch_iterator = AlignedBatchIterator(self.batch_size, shuffle=True)

        batch_iterator_args = [X, Y] + Cs
        if not all_elements_equal_len(batch_iterator_args):
            raise Exception("X, Y and C must have same len!")

        return self._fit([batch_iterator]*2, [batch_iterator_args]*2, 
                         "standard", X_val, y_val,)
        
    def fit_XC_XY(self, X1, Cs, X2, Y,
            batch_iterator=None, 
            batch_iterator_XC=None,
            X_val=None, y_val=None):

        if not isiterable(Cs) or len(Cs[0]) != len(X1):
            Cs = [Cs]

        if batch_iterator is None:
            batch_iterator = AlignedBatchIterator(self.batch_size, shuffle=True)

        batch_iterators = []
        if batch_iterator_XC is not None:
            batch_iterators.append(batch_iterator_XC)
        else:
            batch_iterators.append(batch_iterator)
        # second iterator is always batch_iterator
        batch_iterators.append(batch_iterator)
        
        batch_iterator_args_lst = [ [X1] + Cs, [X2, Y] ]
        if not all_elements_equal_len(batch_iterator_args_lst[0]):
            raise Exception("X1 and all entries in Cs must have same len!")
        if not all_elements_equal_len(batch_iterator_args_lst[1]):
            raise Exception("X2 and Y must have same len!")
            
        return self._fit(batch_iterators, batch_iterator_args_lst, 
                         "alternating", X_val, y_val)
        
    def _fit(self, batch_iterators, batch_iterator_args_lst, simultaneous_mode="standard", X_val=None, y_val=None):

        assert (simultaneous_mode in ["standard", "alternating"])

        if X_val is not None and y_val is not None:
            assert (len(X_val) == len(y_val))

        # training procedures
        if self.verbose:
            print ("Training procedure: %s" % self.procedure)

        if self.procedure in ['decoupled', 'pretrain_finetune']:
            # first training phase
            if self.verbose:
                print (" Optimize phi & beta using the contextual objective")
            train_fn = self._compile_train_fn(self.pattern.training_input_vars,
                                              loss_weights={'target_weight': 0.0, 'context_weight': 1.0}, 
                                              tags= {'psi': False} )
            self._train(train_fn, self.val_fn, batch_iterators[0], batch_iterator_args_lst[0], X_val, y_val)

            # second training phase
            if self.procedure == 'decoupled':
                if self.verbose:
                    print (" Optimize psi using the target objective")
                train_fn = self._compile_train_fn(self.pattern.training_input_vars,
                                                  loss_weights={'target_weight': 1.0, 'context_weight': 0.0}, 
                                                  tags= {'psi': True} ) # beta: False?
                self._train(train_fn, batch_iterators[1], batch_iterator_args_lst[1], X_val, y_val)
            elif self.procedure == 'pretrain_finetune':
                if self.verbose:
                    print (" Optimize phi & psi using the target objective")
                train_fn = self._compile_train_fn( loss_weights={'target_weight': 1.0, 'context_weight': 0.0}, 
                                                        tags= {'beta': False} )
                self._train(train_fn, batch_iterators[1], batch_iterator_args_lst[1], X_val, y_val)
        
        elif self.procedure == 'simultaneous':
                if self.verbose:
                    print (" Optimize phi & psi & beta using a weighted sum of target and contextual objective")
                if simultaneous_mode == "standard":
                    print ("   -> standard mode with single training function")
                    train_fn = self._compile_train_fn(self.pattern.training_input_vars,
                                                      loss_weights=self.loss_weights,
                                                      tags= {} )
                    train_fn = [train_fn]*2
                else:
                    print ("   -> alternating mode with two training functions")
                    lw1 = copy.copy(self.loss_weights)
                    lw1['target_weight'] = 0.
                    train_fn1 = self._compile_train_fn([self.pattern.input_var] + list(self.pattern.context_vars),
                        loss_weights=lw1, 
                        tags= {'psi': False} ) 

                    lw2 = copy.copy(self.loss_weights)
                    lw2['context_weight'] = 0.
                    train_fn2 = self._compile_train_fn([self.pattern.input_var, self.pattern.target_var],
                        loss_weights=lw2, 
                        tags= {'beta': False} )

                    train_fn = [train_fn1, train_fn2]
                    
                self._train(train_fn, batch_iterators, batch_iterator_args_lst, simultaneous_mode, X_val, y_val)
        
        return self

    def _train(self, train_fns, batch_iterators, batch_iterator_args_lst, simultaneous_mode, X_val=None, y_val=None):
        # we switch of the on_unused_input warning because in various
        # circumstance, e.g. in the decoupled procedure, we might pass
        # unused inputs to the training function (e.g. target_var
        # is not used in the pretrain phase). The only reason for having this
        # so we don't have to deal with implementing different 
        # training_input_vars methods in the Pattern class
        theano.config.on_unused_input = 'ignore'

        assert (simultaneous_mode in ["standard", "alternating"])
        
        if self.val_fn is None:
            self.val_fn = self._compile_val_fn()
        
        if not isiterable(train_fns):
            train_fns = [train_fns]
        if not isiterable(batch_iterators):
            batch_iterators = [batch_iterators]
        if not isiterable(batch_iterator_args_lst):
            batch_iterator_args_lst = [batch_iterator_args_lst]
        
        if self.verbose:
            print("Starting training...")
        # We iterate over epochs:
        for epoch in range(self.num_epochs):
            # In each epoch, we do a full pass over the training data:
            train_err = 0
            train_batches = 0
            start_time = time.time()
            
            for train_fn, batch_iterator, batch_iterator_args \
                in zip(train_fns, batch_iterators, batch_iterator_args_lst):
                for batch in batch_iterator(*batch_iterator_args):
                    train_err += train_fn(*batch)
                    train_batches += 1
            train_batches /= len(train_fns)

            # And a pass over the validation data:
            if X_val is not None and y_val is not None:
                val_err, val_acc = self.score(X_val, y_val, batch_size=len(X_val))
    
            # Then we print the results for this epoch:
            if self.verbose:
                print("Epoch {} of {} took {:.3f}s".format(
                    epoch + 1, self.num_epochs, time.time() - start_time))
                print("  training loss:\t\t{:.6f}".format(train_err / train_batches))

                if X_val is not None and y_val is not None:
                    print("  validation loss:\t\t{:.6f}".format(val_err))
                    print("  validation accuracy:\t\t{:.2f} %".format(val_acc * 100))
                    
    def score(self, X, y, batch_size=None, verbose=False):
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
                err = res
                acc = 0
            test_err += err
            test_acc += acc
            test_batches += 1
    
        if test_batches > 0 and verbose:
            print("Score:")
            print("  loss:\t\t\t{:.6f}".format(test_err / test_batches))
            print("  accuracy:\t\t{:.2f} %".format(
                test_acc / test_batches * 100))                

        return test_err/test_batches, test_acc/test_batches
        