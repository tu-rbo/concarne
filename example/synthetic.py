# -*- coding: utf-8 -*-

#!/usr/bin/env python

"""
Usage example for 

This example corresponds to the "synthetic data" experiment in the paper
[http://arxiv.org/abs/1511.06429], illustrating the direct pattern.

Note that this example is structured very similar to the mnist example for
Lasagne in order to facilitate usage of concarne for Lasagne users.
"""

from __future__ import print_function

import concarne
import concarne.patterns

import lasagne
import theano
import theano.tensor as T

import numpy as np

import argparse
import os
import sys
import time

# ################## Download and prepare the synthetic dataset ##################
def load_dataset(tr_data, tr_data_url, test_data, test_data_url):
    # We first define a download function, supporting both Python 2 and 3.
    if sys.version_info[0] == 2:
        from urllib import urlretrieve
    else:
        from urllib.request import urlretrieve

    def download(fname, url):
        print("Downloading %s" % url)
        urlretrieve(url, fname)

    if not os.path.exists(tr_data):
        download(tr_data, tr_data_url)
    if not os.path.exists(test_data):
        download(test_data, test_data_url)

    npz_train = np.load(tr_data)
    npz_test = np.load(test_data)

    X = npz_train['X']
    Y = np.cast['int32'](npz_train['Y'])
    C = npz_train['C']

    X_valid = npz_train['X_valid']
    Y_valid = np.cast['int32'](npz_train['Y_valid'])
    #C_valid = npz_train['C_valid']

    X_test = npz_test['X_test']
    Y_test = np.cast['int32'](npz_test['Y_test'])
    #C_test = npz_train['C_test']
    
    return X, Y, C, X_valid, Y_valid, X_test, Y_test
    
def load_direct_context_dataset():
    tr_data = "cl_synth_direct_d-50_e-0_n-500_seed-12340.npz"
    tr_data_url = "https://tubcloud.tu-berlin.de/public.php?service=files&t=5306a60ec558d8f1efbefaa9438a7261&download"
    test_data = "cl_synth_direct_d-50_e-0_ntest-50000_seed-12340.npz"
    test_data_url = "https://tubcloud.tu-berlin.de/public.php?service=files&t=de69be9c5194defb985166b93f93d017&download"
    return load_dataset(tr_data, tr_data_url, test_data, test_data_url)

def load_embedding_context_dataset():
    tr_data = "cl_synth_direct_d-50_e-0_n-500_seed-12340.npz"
    tr_data_url = "https://tubcloud.tu-berlin.de/public.php?service=files&t=5306a60ec558d8f1efbefaa9438a7261&download"
    test_data = "cl_synth_direct_d-50_e-0_ntest-50000_seed-12340.npz"
    test_data_url = "https://tubcloud.tu-berlin.de/public.php?service=files&t=de69be9c5194defb985166b93f93d017&download"
    return load_dataset(tr_data, tr_data_url, test_data, test_data_url)

def load_relative_context_dataset():
    tr_data = "cl_synth_direct_d-50_e-0_n-500_seed-12340.npz"
    tr_data_url = "https://tubcloud.tu-berlin.de/public.php?service=files&t=5306a60ec558d8f1efbefaa9438a7261&download"
    test_data = "cl_synth_direct_d-50_e-0_ntest-50000_seed-12340.npz"
    test_data_url = "https://tubcloud.tu-berlin.de/public.php?service=files&t=de69be9c5194defb985166b93f93d017&download"
    return load_dataset(tr_data, tr_data_url, test_data, test_data_url)


# ############################# Helper functions #################################
def build_linear_simple(input_layer, n_out, nonlinearity=None, name=None):
    network = lasagne.layers.DenseLayer(input_layer, n_out, nonlinearity=nonlinearity, b=None, name=name)
    return network    


#  ########################## Build Direct Pattern ###############################
def build_direct_pattern(input_var, target_var, context_var, n, m, num_classes):
    input_layer = lasagne.layers.InputLayer(shape=(None, n),
                                        input_var=input_var)
    phi = build_linear_simple( input_layer, m, name="phi")
    psi = build_linear_simple( phi, num_classes, 
        nonlinearity=lasagne.nonlinearities.softmax, name="psi")
    
    # if you want to change the standard loss terms used by a pattern
    # you can define them here and pass them to the Pattern object
    #target_loss=lasagne.objectives.categorical_crossentropy(
    #    psi.get_output_for(phi.get_output_for(input_var)), 
    #    target_var)    
    #context_loss=lasagne.objectives.squared_error(
    #    phi.get_output_for(input_var), 
    #    context_var)
        
    dp = concarne.patterns.DirectPattern(phi=phi, psi=psi, 
                                         target_var=target_var, 
                                         context_var=context_var,
                                         #target_loss=target_loss.mean(),
                                         #context_loss=context_loss.mean()
                                         )
    return dp                                         

def iterate_minibatches(inputs, targets, batchsize, contexts=None, shuffle=False):
    """ Simple iterator for direct pattern """
    assert len(inputs) == len(targets)
    indices = np.arange(len(inputs))

    if shuffle:
        np.random.shuffle(indices)
    
    for start_idx in range(0, len(inputs) - batchsize + 1, batchsize):
        excerpt = indices[start_idx:start_idx + batchsize]
        if contexts is None:
            yield inputs[excerpt], targets[excerpt]
        else:
            yield inputs[excerpt], targets[excerpt], contexts[excerpt]


#  ########################## Build Pairwise Pattern ###############################





#  ########################## Main ###############################

def main(data, num_epochs=500, batchsize=50):
    #theano.config.on_unused_input = 'ignore'
    
    # Prepare Theano variables for inputs and targets
    input_var = T.matrix('inputs')
    target_var = T.ivector('targets')
    context_var = T.matrix('contexts')
    
    pattern = None
    if data == "direct":
        # Load the dataset
        print("Loading data...")
        X_train, y_train, C_train, X_val, y_val, X_test, y_test = load_direct_context_dataset()
    
        # input dimension of X
        n = X_train.shape[1]
        # intermediate dimension of C
        m = C_train.shape[1]
        # number of classes in example
        num_classes = 2
        pattern = build_direct_pattern(input_var, target_var, context_var, n, m, num_classes)

    # Get the loss expression for training
    loss = pattern.training_loss()
    loss = loss.mean()

    # Create update expressions for training, i.e., how to modify the
    # parameters at each training step. Here, we'll use Stochastic Gradient
    # Descent (SGD) with Nesterov momentum, but Lasagne offers plenty more.
    params = lasagne.layers.get_all_params(pattern, trainable=True)
    #params = dp.get_all_params(trainable=True)
    updates = lasagne.updates.nesterov_momentum(
            loss, params, learning_rate=0.0001, momentum=0.9)


    # Create a loss expression for validation/testing. The crucial difference
    # here is that we do a deterministic forward pass through the network,
    # disabling dropout layers.
    test_prediction = lasagne.layers.get_output(pattern, deterministic=True)
    test_loss = lasagne.objectives.categorical_crossentropy(test_prediction,
                                                            target_var)
    test_loss = test_loss.mean()
    # As a bonus, also create an expression for the classification accuracy:
    test_acc = T.mean(T.eq(T.argmax(test_prediction, axis=1), target_var),
                      dtype=theano.config.floatX)

    # Compile a function performing a training step on a mini-batch (by giving
    # the updates dictionary) and returning the corresponding training loss:
    train_fn = theano.function([input_var, target_var, context_var], loss, updates=updates)

    # Compile a second function computing the validation loss and accuracy:
    val_fn = theano.function([input_var, target_var], [test_loss, test_acc])

    # Finally, launch the training loop.
    print("Starting training...")
    # We iterate over epochs:
    for epoch in range(num_epochs):
        # In each epoch, we do a full pass over the training data:
        train_err = 0
        train_batches = 0
        start_time = time.time()
        for batch in iterate_minibatches(X_train, y_train, batchsize, C_train, shuffle=True):
            inputs, targets, contexts = batch
            train_err += train_fn(inputs, targets, contexts)
            train_batches += 1

        # And a full pass over the validation data:
        val_err = 0
        val_acc = 0
        val_batches = 0
        for batch in iterate_minibatches(X_val, y_val, batchsize, shuffle=False):
            inputs, targets = batch
            err, acc = val_fn(inputs, targets)
            val_err += err
            val_acc += acc
            val_batches += 1

        # Then we print the results for this epoch:
        print("Epoch {} of {} took {:.3f}s".format(
            epoch + 1, num_epochs, time.time() - start_time))
        print("  training loss:\t\t{:.6f}".format(train_err / train_batches))
        print("  validation loss:\t\t{:.6f}".format(val_err / val_batches))
        print("  validation accuracy:\t\t{:.2f} %".format(
            val_acc / val_batches * 100))

    # After training, we compute and print the test error:
    test_err = 0
    test_acc = 0
    test_batches = 0
    for batch in iterate_minibatches(X_test, y_test, 500, shuffle=False):
        inputs, targets = batch
        err, acc = val_fn(inputs, targets)
        test_err += err
        test_acc += acc
        test_batches += 1
    print("Final results:")
    print("  test loss:\t\t\t{:.6f}".format(test_err / test_batches))
    print("  test accuracy:\t\t{:.2f} %".format(
        test_acc / test_batches * 100))
        
        
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("data", type=str, help="which data to select", default='direct', choices=['direct', 'embedding', 'pairwise'])
    parser.add_argument("--num_epochs", type=int, help="number of epochs for SGD", default=500, required=False)
    parser.add_argument("--batchsize", type=int, help="batch size for SGD", default=50, required=False)
    args = parser.parse_args()
  
    main(args.data, args.num_epochs, args.batchsize)
