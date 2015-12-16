# -*- coding: utf-8 -*-

from __future__ import print_function

import concarne.iterators

import lasagne
import theano
import theano.tensor as T
import time


class PatternTrainer(object):
    
    def __init__(self, pattern, 
                 num_epochs, 
                 learning_rate,
                 procedure="simultaneous", 
                 target_weight=None, 
                 context_weight=None):
        self.pattern = pattern
        self.num_epochs = num_epochs
        self.learning_rate = learning_rate
        self.procedure = procedure
        assert (procedure in ["simultaneous", "decoupled", "pretrain_finetune"])
        
        # target_weight and context_weight are only relevant for simultaneous
        self.loss_weights = {}
        if self.procedure == "simultaneous":
            if self.target_weight is not None:
                self.loss_weights['target_weight'] = self.target_weight
            if self.context_weight is not None:
                self.loss_weights['context_weight'] = self.context_weight
        
        # simple iterator for valid/test error
        self.aligned_batch_iterator = \
            concarne.iterators.AlignedBatchIterator(self.batch_iterator.batch_size, shuffle=False)
        
        self.train_fn = None
        self.val_fn = None

    def _compile_train_fn(self, loss_weights, tags):
        # TODO where to get the train fn inputs from? from the pattern?
        #  but then in fact we should also get the 
        train_fn_inputs = None
        
        loss = self.pattern.training_loss(**loss_weights).mean()
        
        params = lasagne.layers.get_all_params(self.pattern, trainable=True, **tags)
        
        # TODO add possibility to use different update mechanism
        updates = lasagne.updates.nesterov_momentum(
            loss, params, learning_rate=self.learning_rate, momentum=0.9)
    
        # Compile a function performing a training step on a mini-batch (by giving
        # the updates dictionary) and returning the corresponding training loss:
        train_fn = theano.function(train_fn_inputs, loss, updates=updates)
    
        return train_fn

    def _compile_val_fn(self):
        input_var = self.pattern.input_var
        target_var = self.pattern.target_var
        
        # TODO add possibility to use different test los
        test_prediction = lasagne.layers.get_output(self.pattern, deterministic=True)
        test_loss = lasagne.objectives.categorical_crossentropy(test_prediction,
                                                                target_var).mean()
        # As a bonus, also create an expression for the classification accuracy:
        test_acc = T.mean(T.eq(T.argmax(test_prediction, axis=1), target_var),
                          dtype=theano.config.floatX)
    
        # Compile a second function computing the validation loss and accuracy:
        val_fn = theano.function([input_var, target_var], [test_loss, test_acc])

        return val_fn
        
    def _train(self, batch_iterator_args, num_epochs, X_val=None, y_val=None):
        train_fn = self.train_fn
        val_fn = self.val_fn
        
        print("Starting training...")
        # We iterate over epochs:
        for epoch in range(num_epochs):
            # In each epoch, we do a full pass over the training data:
            train_err = 0
            train_batches = 0
            start_time = time.time()
            for batch in self.batch_iterator(**batch_iterator_args):
                train_err += train_fn(*batch)
                train_batches += 1
    
            # And a full pass over the validation data:
            if X_val is not None and y_val is not None:
                val_err = 0
                val_acc = 0
                val_batches = 0
                for batch in self.simple_batch_iterator(X_val, y_val):
                    inputs, targets = batch
                    err, acc = val_fn(inputs, targets)
                    val_err += err
                    val_acc += acc
                    val_batches += 1
    
            # Then we print the results for this epoch:
            print("Epoch {} of {} took {:.3f}s".format(
                epoch + 1, self.num_epochs, time.time() - start_time))
            print("  training loss:\t\t{:.6f}".format(train_err / train_batches))

            if X_val is not None and y_val is not None:
                print("  validation loss:\t\t{:.6f}".format(val_err / val_batches))
                print("  validation accuracy:\t\t{:.2f} %".format(
                    val_acc / val_batches * 100))
    
    def fit(self, batch_iterator_args, num_epochs=None, X_val=None, y_val=None):
        if num_epochs is None:
            num_epochs = self.num_epochs

        valid_keys = self.batch_iterator.get_keys()
        for k in batch_iterator_args.keys():
            if k not in valid_keys:
                raise Exception("Not a valid key '%s' for iterator %s (allowed keys: %s)" \
                    % (k, self.iterator.__class__.__name__, ", ".join(valid_keys)))

        if self.val_fn is None:
            self.val_fn = self._compile_val_fn()

        # training procedures
        print ("Training procedure: %s" % self.procedure)

        if self.procedure in ['decoupled', 'pretrain_finetune']:
            # first training phase
            print (" Optimize phi & beta using the contextual objective")
            self.train_fn = self._compile_train_fn( loss_weights={'target_weight': 0.0, 'context_weight': 1.0}, 
                                                    tags= {'psi': False} )
            self._train(batch_iterator_args, num_epochs, X_val, y_val)

            # second training phase
            if self.procedure == 'decoupled':
                print (" Optimize psi using the target objective")
                self.train_fn = self._compile_train_fn( loss_weights={'target_weight': 1.0, 'context_weight': 0.0}, 
                                                        tags= {'psi': True} ) # beta: False?
                self._train(batch_iterator_args, num_epochs, X_val, y_val)
            elif self.procedure == 'pretrain_finetune':
                print (" Optimize phi & psi using the target objective")
                self.train_fn = self._compile_train_fn( loss_weights={'target_weight': 1.0, 'context_weight': 0.0}, 
                                                        tags= {'beta': False} )
                self._train(batch_iterator_args, num_epochs, X_val, y_val)
        
        elif self.procedure == 'simultaneous':
                print (" Optimize phi & psi & beta using a weighted sum of target and contextual objective")
                self.train_fn = self._compile_train_fn( loss_weights=self.loss_weights,
                                                        tags= {} )
                self._train(batch_iterator_args, num_epochs, X_val, y_val)
        
        return self.pattern

    def score(self, X, y, batch_size=None):
        if self.val_fn in None:
            self.val_fn = self._compile_val_fn()
        
        # After training, we compute and print the test error:
        test_err = 0
        test_acc = 0
        test_batches = 0
        for batch in self.simple_batch_iterator(X, y):
            inputs, targets = batch
            err, acc = self.val_fn(inputs, targets)
            test_err += err
            test_acc += acc
            test_batches += 1
    
        if test_batches > 0:
            print("Final results:")
            print("  test loss:\t\t\t{:.6f}".format(test_err / test_batches))
            print("  test accuracy:\t\t{:.2f} %".format(
                test_acc / test_batches * 100))                

        