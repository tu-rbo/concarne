# -*- coding: utf-8 -*-

#!/usr/bin/env python

"""
This example corresponds use the "synthetic data" experiment in the paper
http://arxiv.org/abs/1511.06429 to illustrate how to use the pattern trainer.

Note that the file is structured very similarly to the MNIST example for
Lasagne in order to facilitate usage of concarne for Lasagne users.

In order to run the example first install concarne, or run it from the 
repository root with the command
 python -m example/pattern_trainer_synthetic <parameters>

The script accepts a couple of parameters, e.g.
  python pattern_trainer_synthetic.py multiview direct  --num_epochs 500 --batchsize 50
  
Run the script with 
  python pattern_trainer_synthetic.py --help
for details

"""

from __future__ import print_function

import concarne
import concarne.patterns
import concarne.iterators
import concarne.training

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

    # sanity check
    assert (np.mean (npz_train['R'] - npz_test['R']) == 0.)

    X = npz_train['X']
    Y = np.cast['int32'](npz_train['Y'])
    C = npz_train['C']

    X_valid = npz_train['X_valid']
    Y_valid = np.cast['int32'](npz_train['Y_valid'])
    #C_valid = npz_train['C_valid']

    X_test = npz_test['X_test']
    Y_test = np.cast['int32'](npz_test['Y_test'])
    #C_test = npz_train['C_test']
    
    return (npz_train, npz_test),\
      (X, Y, C, X_valid, Y_valid, X_test, Y_test)
    
def load_direct_context_dataset():
    tr_data = "cl_synth_direct_d-50_e-0_n-500_seed-12340.npz"
    tr_data_url = "https://tubcloud.tu-berlin.de/public.php?service=files&t=865193384483af385172f5871aa5cd36&path=%2Fsynthetic_data%2Fdirect&files=cl_synth_direct_d-50_e-0_n-500_seed-12340.npz&download"
    test_data = "cl_synth_direct_d-50_e-0_ntest-50000_seed-12340.npz"
    test_data_url = "https://tubcloud.tu-berlin.de/public.php?service=files&t=865193384483af385172f5871aa5cd36&path=%2Fsynthetic_data%2Fdirect&files=cl_synth_direct_d-50_e-0_ntest-50000_seed-12340.npz&download"
    return load_dataset(tr_data, tr_data_url, test_data, test_data_url)[1]

def load_embedding_context_dataset():
    tr_data = "cl_synth_embedding_d-50_e-25_n-500_seed-12340.npz"
    tr_data_url = "https://tubcloud.tu-berlin.de/public.php?service=files&t=865193384483af385172f5871aa5cd36&path=%2Fsynthetic_data%2Fembedding&files=cl_synth_embedding_d-50_e-25_n-500_seed-12340.npz&download"
    test_data = "cl_synth_embedding_d-50_e-25_ntest-50000_seed-12340.npz"
    test_data_url = "https://tubcloud.tu-berlin.de/public.php?service=files&t=865193384483af385172f5871aa5cd36&path=%2Fsynthetic_data%2Fembedding&files=cl_synth_embedding_d-50_e-25_ntest-50000_seed-12340.npz&download"

    (npz_train, npz_test), res = \
      load_dataset(tr_data, tr_data_url, test_data, test_data_url)
      
    # sanity check
    assert (np.mean (npz_train['Q'] - npz_test['Q']) == 0.)
      
    return res

def load_relative_context_dataset():
    tr_data = "cl_synth_relative_d-50_e-0_n-500_seed-12340.npz"
    tr_data_url = "https://tubcloud.tu-berlin.de/public.php?service=files&t=865193384483af385172f5871aa5cd36&path=%2Fsynthetic_data%2Frelative&files=cl_synth_relative_d-50_e-0_n-500_seed-12340.npz&download"
    test_data = "cl_synth_relative_d-50_e-0_ntest-50000_seed-12340.npz"
    test_data_url = "https://tubcloud.tu-berlin.de/public.php?service=files&t=865193384483af385172f5871aa5cd36&path=%2Fsynthetic_data%2Frelative&files=cl_synth_relative_d-50_e-0_ntest-50000_seed-12340.npz&download"

    (_,_), (X, Y, C, X_valid, Y_valid, X_test, Y_test) \
       = load_dataset(tr_data, tr_data_url, test_data, test_data_url)
    
    # the context training data C contains stacked "x_j" and "y_ij" 
    # which are aligned with the x_i in matrix X
    CX = C[:, :X.shape[1]]
    CY = C[:, X.shape[1]:]
    
    return X, Y, CX, CY, X_valid, Y_valid, X_test, Y_test


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
    #   target_loss=lasagne.objectives.categorical_crossentropy(
    #       psi.get_output_for(phi.get_output_for(input_var)), 
    #       target_var)    
    #   context_loss=lasagne.objectives.squared_error(
    #       phi.get_output_for(input_var), 
    #       context_var)
        
    # alternatively - and even easier - you can also just pass the lasagne
    # objective function and the pattern will automatically figure out
    # inputs and outputs:
    #  target_loss=lasagne.objectives.categorical_crossentropy,
    #  context_loss=lasagne.objectives.squared_error
        
    dp = concarne.patterns.DirectPattern(phi=phi, psi=psi, 
                                         target_var=target_var, 
                                         context_var=context_var,
                                         target_loss=lasagne.objectives.categorical_crossentropy,
                                         context_loss=lasagne.objectives.squared_error
                                         )
    return dp

#  ########################## Build Multi-task Pattern ###############################
def build_multitask_pattern(input_var, target_var, context_var, n, m, d, num_classes):
    input_layer = lasagne.layers.InputLayer(shape=(None, n),
                                        input_var=input_var)
    phi = build_linear_simple( input_layer, d, name="phi")
    psi = build_linear_simple( phi, num_classes, 
        nonlinearity=lasagne.nonlinearities.softmax, name="psi")
    beta = build_linear_simple( phi, m, name="beta")
        
    mtp = concarne.patterns.MultiTaskPattern(phi=phi, psi=psi, beta=beta,
                                         target_var=target_var, 
                                         context_var=context_var,
                                         context_loss=lasagne.objectives.squared_error,
                                         )
    return mtp
    
#  ########################## Build Multi-view Pattern ###############################
def build_multiview_pattern(input_var, target_var, context_var, n, m, d, num_classes):
    input_layer = lasagne.layers.InputLayer(shape=(None, n),
                                        input_var=input_var)
    context_input_layer = lasagne.layers.InputLayer(shape=(None, m),
                                        input_var=context_var)
    phi = build_linear_simple( input_layer, d, name="phi")
    psi = build_linear_simple( phi, num_classes, 
        nonlinearity=lasagne.nonlinearities.softmax, name="psi")
    beta = build_linear_simple( context_input_layer, d, name="beta")
        
    mtp = concarne.patterns.MultiViewPattern(phi=phi, psi=psi, beta=beta,
                                         target_var=target_var, 
                                         context_var=context_var,
                                         context_loss=lasagne.objectives.squared_error,
                                         )
    return mtp    
    
#  ########################## Build Pairwise Pattern ###############################

def build_pw_transformation_pattern(input_var, target_var, context_var, context_transform_var, n, m, num_classes):
    input_layer = lasagne.layers.InputLayer(shape=(None, n),
                                        input_var=input_var)
    phi = build_linear_simple( input_layer, m, name="phi")
    psi = build_linear_simple( phi, num_classes, 
        nonlinearity=lasagne.nonlinearities.softmax, name="psi")
    
    # optionally, we can also learn parameters for beta.
    # here it does not make much sense because all transformations
    # are linear.
    #beta = build_linear_simple( phi, m, name="beta")
    
    # otherwise, we just set beta=None, which will make beta the identity
    beta = None
        
    pptp = concarne.patterns.PairwisePredictTransformationPattern(phi=phi, psi=psi, 
                                         beta=beta,
                                         target_var=target_var, 
                                         context_var=context_var,
                                         context_transform_var=context_transform_var,
                                         )
    return pptp


#  ########################## Main ###############################

def main(pattern_type, data, procedure, num_epochs=500, batchsize=50):
#if __name__ == "__main__":
#    pattern_type="multiview"
#    data='direct'
#    num_epochs=500
#    batchsize=50
    #theano.config.on_unused_input = 'ignore'

    print ("Pattern: %s" % pattern_type)

    if data in ["direct", "embedding"]:
      assert (pattern_type != "pairwise")
    elif data == "relative":
      assert (pattern_type == "pairwise")
    else:
      raise Exception("Unsupported data %s" % data)
    
    # ------------------------------------------------------
    # Prepare Theano variables for inputs and targets
    input_var = T.matrix('inputs')
    target_var = T.ivector('targets')
    context_var = T.matrix('contexts')
    
    # number of classes in example
    num_classes = 2
    
    pattern = None
    momentum = 0.9
        
    # ------------------------------------------------------
    # Load data and build pattern
    if data == "direct":
        print("Loading direct data...")
        X_train, y_train, C_train, X_val, y_val, X_test, y_test = load_direct_context_dataset()
    
    if data == "embedding":    
        print("Loading embedding data...")
        X_train, y_train, C_train, X_val, y_val, X_test, y_test = load_embedding_context_dataset()
    
    if data == "direct" or data == "embedding":
        # input dimension of X
        n = X_train.shape[1]
        # dimensionality of C
        m = C_train.shape[1]
        # dimensionality of intermediate representation S
        d = 1
        
        if pattern_type == "direct":
          # d == m
          pattern = build_direct_pattern(input_var, target_var, context_var, n, m, num_classes)
          learning_rate=0.0001
          if procedure != "simultaneous":
            learning_rate*=0.1
          loss_weights = {} #'target_weight':0.5, 'context_weight':0.5}

        elif pattern_type == "multitask":
          pattern = build_multitask_pattern(input_var, target_var, context_var, n, m, d, num_classes)
          learning_rate=0.0001
          if procedure != "simultaneous":
            learning_rate*=0.1
          loss_weights = {'target_weight':0.9, 'context_weight':0.1}

        elif pattern_type == "multiview":
          pattern = build_multiview_pattern(input_var, target_var, context_var, n, m, d, num_classes)
          learning_rate=0.001
          if procedure != "simultaneous":
            learning_rate*=0.01
          loss_weights = {'target_weight':0.99, 'context_weight':0.01}
          
        iterate_context_minibatches_args = [X_train, y_train, C_train]
    
        
    elif data == "relative":
        # Load the dataset
        print("Loading relative data...")
        X_train, y_train, CX_train, Cy_train, X_val, y_val, X_test, y_test = load_relative_context_dataset()

        context_transform_var = T.matrix('context_transforms')
    
        # input dimension of X
        n = X_train.shape[1]
        # intermediate dimension of C
        m = Cy_train.shape[1]

        pattern = build_pw_transformation_pattern(input_var, target_var, context_var, context_transform_var, n, m, num_classes)

        iterate_context_minibatches_args = [X_train, y_train, [CX_train, Cy_train]]
        #train_fn_inputs = [input_var, target_var, context_var, context_transform_var]
        
        learning_rate=0.0001        
        loss_weights = {'target_weight':0.1, 'context_weight':0.9}
    
    # ------------------------------------------------------
    # Get the loss expression for training
    
    trainer = concarne.training.PatternTrainer(pattern,
                                               num_epochs,
                                               learning_rate,
                                               batchsize,
                                               momentum,
                                               procedure,
                                               verbose=True,
                                               **loss_weights
                                               )
    print("Starting training...")
    trainer.fit_XYC(*iterate_context_minibatches_args, X_val=X_val, y_val=y_val)

    print("=================")
    print("Test score...")
    trainer.score(X_test, y_test, verbose=True)
        
    return pattern
        
# ------------------------------------------------------        
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("pattern",  nargs='?', type=str, help="which pattern to use", 
                        default='direct', 
                        choices=['direct', 'multitask', 'multiview', 'pairwise'])
    parser.add_argument("data",  nargs='?', type=str, help="which context data to load", 
                        default='direct', 
                        choices=['direct', 'embedding', 'relative'])
    parser.add_argument("procedure", nargs='?', type=str, help="training procedure", 
                        default='simultaneous', choices=['decoupled', 'simultaneous', 'pretrain_finetune'])
    parser.add_argument("--num_epochs", type=int, help="number of epochs for SGD", default=500, required=False)
    parser.add_argument("--batchsize", type=int, help="batch size for SGD", default=50, required=False)
    args = parser.parse_args()
  
    pattern = main(args.pattern, args.data, args.procedure, args.num_epochs, args.batchsize)
