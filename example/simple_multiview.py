# -*- coding: utf-8 -*-

"""
This example illustrates how simple it is to train a classifier using
side information.

It illustrates the exemplary use of the multi-view pattern; for more info
on how to use other patterns, check out synthetic.py.

For a realistic example with real data check out handwritten.py. 

For more details see the documentation and the paper
http://arxiv.org/abs/1511.06429 
"""

from __future__ import print_function

import concarne
import concarne.patterns
import concarne.training

import lasagne
import theano.tensor as T

try:
    import sklearn.linear_model as sklm
except:
    print (
"""You don't have scikit-learn installed; install it to compare
learning with side information to simple supervised learning""")
    sklm = None

import numpy as np



if __name__ == "__main__":
    
    #--------------------------------------------------------    
    # Generate the data
    
    num_samples = 300
    
    input_dim = 50
    side_dim = 50
    
    # generate some random data with 100 samples and 5 dimensions
    X = np.random.randn(num_samples, input_dim)
    
    # select the third dimension as the relevant for our classification
    # task
    S = X[:, 2:3]
    
    # The labels are simply the sign of S 
    # (note the downcast to int32 - this is required by theano)
    y = np.asarray(S > 0, dtype='int32').reshape( (-1,) )
    # This means we have 2 classes - we will use that later for building
    # the pattern
    num_classes = 2
    
    # Now let's define some side information: we simulate an additional sensor
    # which contains S, but embedded into a different space
    C = np.random.randn(num_samples, side_dim)
    # set second dimension of C to correspond to S
    C[:, 1] = S[:,0]
    
    # let's make it harder to find S in X and C by applying a random rotations
    # to both data sets
    R = np.linalg.qr(np.random.randn(input_dim, input_dim))[0] # random linear rotation
    X = X.dot(R)

    Q = np.linalg.qr(np.random.randn(side_dim, side_dim))[0] # random linear rotation
    C = C.dot(Q)
   
    #--------------------------------------------------------
    # Define the pattern
    
    # now that we have some data, we can use a pattern to learn
    # from it. 
    # since X and C are two different "views" of the relevant data S,
    # the multi-view pattern is the most natural choice.
    
    # The pattern needs three functions: phi(X) which maps X to an intermediate
    # representation (that should somewhat correspond to S); psi which 
    # performs classification using phi(X); and beta(C) which maps C to S.
    # The goal of the multi-view pattern is to find phi and beta, s.t.
    # phi(X)~beta(X) and psi s.t. that psi(phi(X))~Y
    
    # Let's first define the theano variables which will represent our data
    input_var = T.matrix('inputs')  # for X
    target_var = T.ivector('targets')  # for Y
    side_var = T.matrix('sideinfo')  # for C
    
    # Size of the intermediate representation phi(X); since S is 1-dim,
    # phi(X) can also map to a 1-dim vector
    representation_dim = 1 

    # Now define the functions - we choose linear functions
    # there are two ways to do it. the first way is to define the 
    # lasagne network (in our case only on layer) yourself.
#    phi = lasagne.layers.DenseLayer(input_layer, representation_dim, nonlinearity=None, b=None, name="phi")

    # the easier way of doing it is to pass a list of tuples with a layer
    # class and the instantion parameters in a dictionary (layer, layer_params).
    # This has the benefit that you don't have to worry about the definition
    # of input layers and the correct  wiring of phi, psi and beta - this is 
    # all taken care of by the pattern.

    # Also, users of the nolearn library might be familiar with this type
    # of specifying a neural network.
        
    # optionally you can pass an input layer, but it is not required and
    # will automatically be inferred by the pattern
    #phi = [ 
        #(lasagne.layers.InputLayer, {'shape': (None, input_dim), 'input_var': input_var}),
        #(lasagne.layers.DenseLayer, { 'num_units': concarne.patterns.PairwisePredictTransformationPattern.PHI_OUTPUT_SHAPE,
        #                             'nonlinearity':None, 'b':None })]
        
    phi = [ (lasagne.layers.DenseLayer, 
             { 
             # for the variable of your layer that denotes the output of the
             # network you should use the markers PHI_OUTPUT_SHAPE,
             # PSI_OUTPUT_SHAPE and BETA_OUTPUT_SHAPE, so that the pattern
             # can automatically infer the correct shape
             'num_units': concarne.patterns.Pattern.PHI_OUTPUT_SHAPE,
             'nonlinearity':None, 'b':None })]
    psi = [(lasagne.layers.DenseLayer, 
            { 'num_units': concarne.patterns.Pattern.PSI_OUTPUT_SHAPE, 
            'nonlinearity':lasagne.nonlinearities.softmax, 'b':None })]
    beta = [(lasagne.layers.DenseLayer, 
            { 'num_units': concarne.patterns.Pattern.BETA_OUTPUT_SHAPE, 
            'nonlinearity':None, 'b':None })]
    
    # now that we have figured our all functions, we can pass them to the pattern
    pattern = concarne.patterns.MultiViewPattern(phi=phi, psi=psi, beta=beta,
                                                 # the following parameters are required to automatically
                                                 # build the functions and the losses
                                                 input_var=input_var, 
                                                 target_var=target_var, 
                                                 side_var=side_var,
                                                 input_shape=input_dim,
                                                 target_shape=num_classes,
                                                 side_shape=side_dim,
                                                 representation_shape=representation_dim,
                                                 # we have to define two loss functions: 
                                                 # the target loss deals with optimizing psi and phi wrt. X & Y
                                                 target_loss=lasagne.objectives.categorical_crossentropy,
                                                 # the side loss deals with optimizing beta and phi wrt. X & C,
                                                 # for multi-view it is beta(C)~phi(X)
                                                 side_loss=lasagne.objectives.squared_error)

    #--------------------------------------------------------
    # Training 
    
    # first split our data into training, test, and validation data
    split = num_samples/3

    X_train = X[:split]
    X_val = X[split:2*split]
    X_test = X[2*split:]
    
    y_train = y[:split]
    y_val = y[split:2*split]
    y_test = y[2*split:]

    C_train = C[:split]
    
    
    # instantiate the PatternTrainer which trains the pattern via stochastic
    # gradient descent
    trainer = concarne.training.PatternTrainer(pattern,
                                               procedure='simultaneous',
                                               num_epochs=200,
                                               batch_size=10,
                                               update=lasagne.updates.nesterov_momentum,
                                               update_learning_rate=0.01,
                                               update_momentum=0.9,
                                               )
   
    # we use the fit_XYC method because our X, Y and C data all have the same
    # size. Also note the [] our C_train - because it is possible to pass
    # multiple side information to some patterns, you have to pass side information
    # in a list.
    # We can also pass validation data to the fit method, however it only
    # has an effect if we set the verbose switch to true to give us
    # information about the learning progress
    trainer.fit_XYZ(X_train, y_train, [C_train], 
                    X_val=X_val, y_val=y_val, 
                    verbose=True)

    # print some statistics
    print("=================")
    print("Test score...")
    trainer.score(X_test, y_test, verbose=True)    
    
    # Let's compare to supervised learning!
    if sklm is not None:
        # let's try different regularizations
        for c in [1e-5, 1e-1, 1, 10, 100, 1e5]:
            lr = sklm.LogisticRegression(C=c)
            lr.fit(X_train, y_train)
            print ("Logistic Regression (C=%f) accuracy = %.3f %%" % (c, 100*lr.score(X_test, y_test)))