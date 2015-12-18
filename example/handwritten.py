#!/usr/bin/env python

"""
This example corresponds to the "handwritten character recognition"
experiment in the paper [http://arxiv.org/abs/1511.06429],
illustrating the various patterns.

Note that the file is structured very similarly to the MNIST example for
Lasagne in order to facilitate usage of concarne for Lasagne users.
"""

from __future__ import print_function
import concarne
import concarne.patterns
import concarne.training
import lasagne
import theano
import theano.tensor as T
import numpy as np
import argparse
import os
import sys
import time
from sklearn import cluster


def min_num_per_label(labels, possible_labels):
    return sum([sum(label == labels) for label in possible_labels])


def split_indices(labels, possible_labels, samples_per_label):
    # if there are less samples in the dataset, only put those in the
    min_samples = min_num_per_label(labels, possible_labels)
    if min_samples <= samples_per_label:
        print("not enough samples to do the requested split")
        samples_per_label = min_samples

    split1, split2 = [], []
    for label in possible_labels:
        # check which samples are of that label
        samples = np.where(labels == label)[0]
        np.random.shuffle(samples)
        split1 += samples[:samples_per_label].tolist()
        split2 += samples[samples_per_label:].tolist()

    # shuffle again (to not have sequences of equal labels)
    np.random.shuffle(split1)
    np.random.shuffle(split2)
    return split1, split2


def apply_split(split1, split2, list_of_arrays):
    return [(array[split1], array[split2]) for array in list_of_arrays]


# ################## Download and prepare the handwritten character recognition dataset ##################
def load_dataset(data_file, data_url):
    # We first define a download function, supporting both Python 2 and 3.
    if sys.version_info[0] == 2:
        from urllib import urlretrieve
    else:
        from urllib.request import urlretrieve

    def download(fname, url):
        print("Downloading %s" % url)
        urlretrieve(url, fname)

    if not os.path.exists(data_file):
        download(data_file, data_url)

    npz = np.load(data_file)

    X = npz['data_im'].astype('float32')
    y = npz['labels'].astype('int32')
    C = npz['data_xy'].astype('float32')
    y_names = npz['label_names']

    num_classes = len(y_names)

    # reshape vectors to images and scale
    X = X.reshape(-1, 1, 32, 32) / np.float32(255) * 2.0 - 1.0

    # split data into training set and rest to have 100 samples per class for training
    split1, split2 = split_indices(y, range(num_classes), 100)
    [(X_train, X_rest), (y_train, y_rest), (C_train, C_rest)] = apply_split(split1, split2, [X, y, C])

    split1, split2 = split_indices(y_train, range(num_classes), 10)
    [(X_sup, _X), (y_sup, _y)] = apply_split(split1, split2, [X_train, y_train])

    # select 10 samples per class from the rest for validation and testing
    split1, split2 = split_indices(y_rest, range(num_classes), 5)
    [(X_val, X_rest), (y_val, y_rest)] = apply_split(split1, split2, [X_rest, y_rest])

    split1, split2 = split_indices(y_rest, range(num_classes), 5)
    [(X_test, X_rest), (y_test, y_rest)] = apply_split(split1, split2, [X_rest, y_rest])

    return npz, (X_train, C_train, X_sup, y_sup, X_val, y_val, X_test, y_test, num_classes)


def load_handwritten_data():
    data_file = 'data.npz'
    data_url = ''
    return load_dataset(data_file, data_url)[1]


def load_handwritten_data_easy():
    data_file = 'data_easy.npz'
    data_url = ''
    return load_dataset(data_file, data_url)[1]


# ############################# Helper functions #################################
def iterate_minibatches(inputs, targets, batchsize, shuffle=False):
    """ Simple iterator for direct pattern """
    assert len(inputs) == len(targets)
    indices = np.arange(len(inputs))

    if shuffle:
        np.random.shuffle(indices)

    for start_idx in range(0, len(inputs) - batchsize + 1, batchsize):
        excerpt = indices[start_idx:start_idx + batchsize]
        yield inputs[excerpt], targets[excerpt]


def build_conv_net(input_var, input_shape, n_out):
    network = lasagne.layers.InputLayer(shape=input_shape, input_var=input_var)
    network = lasagne.layers.Conv2DLayer(network, 32, (5, 5), nonlinearity=lasagne.nonlinearities.rectify)
    network = lasagne.layers.MaxPool2DLayer(network, 2)
    network = lasagne.layers.Conv2DLayer(network, 32, (5, 5), nonlinearity=lasagne.nonlinearities.rectify)
    network = lasagne.layers.MaxPool2DLayer(network, 2)
    network = lasagne.layers.DenseLayer(lasagne.layers.DropoutLayer(network, p=0.5), num_units=n_out,
                                        nonlinearity=lasagne.nonlinearities.rectify)
    return network


def build_view_net(input_var, input_shape, n_out):
    network = lasagne.layers.InputLayer(shape=input_shape, input_var=input_var)
    network = lasagne.layers.DenseLayer(lasagne.layers.DropoutLayer(network, p=0.5), num_units=n_out,
                                        nonlinearity=lasagne.nonlinearities.rectify)
    network = lasagne.layers.DenseLayer(lasagne.layers.DropoutLayer(network, p=0.5), num_units=n_out,
                                        nonlinearity=lasagne.nonlinearities.rectify)
    network = lasagne.layers.DenseLayer(lasagne.layers.DropoutLayer(network, p=0.5), num_units=n_out,
                                        nonlinearity=lasagne.nonlinearities.rectify)
    return network


def build_classifier(network, n_out):
    return lasagne.layers.DenseLayer(lasagne.layers.DropoutLayer(network, p=0.5), num_units=n_out,
                                     nonlinearity=lasagne.nonlinearities.softmax)


def build_regressor(network, n_out):
    return lasagne.layers.DenseLayer(lasagne.layers.DropoutLayer(network, p=0.5), num_units=n_out,
                                     nonlinearity=lasagne.nonlinearities.linear)


# ########################## Build Direct Pattern ###############################
def build_direct_pattern(input_var, target_var, context_var, input_shape, n_hidden, num_classes):
    phi = build_conv_net(input_var, input_shape, n_hidden)
    psi = build_classifier(phi, num_classes)
    return concarne.patterns.DirectPattern(phi=phi, psi=psi, target_var=target_var, context_var=context_var)


# ########################## Build Multi-task Pattern ###############################
def build_multitask_pattern(input_var, target_var, context_var, input_shape, n_hidden, num_classes, n_out_context,
                            discrete=True):
    phi = build_conv_net(input_var, input_shape, n_hidden)
    psi = build_classifier(phi, num_classes)
    if discrete:
        beta = build_classifier(phi, n_out_context)
        context_loss = lasagne.objectives.categorical_crossentropy(lasagne.layers.get_output(beta), context_var).mean()
    else:
        beta = build_regressor(phi, n_out_context)
        context_loss = lasagne.objectives.squared_error(lasagne.layers.get_output(beta), context_var).mean()

    return concarne.patterns.MultiTaskPattern(phi=phi, psi=psi, beta=beta, target_var=target_var,
                                              context_var=context_var, context_loss=context_loss)


#  ########################## Build Multi-view Pattern ###############################
def build_multiview_pattern(input_var, target_var, context_var, input_shape, n_hidden, num_classes):
    phi = build_conv_net(input_var, input_shape, n_hidden)
    psi = build_classifier(phi, num_classes)
    beta = build_view_net(input_var, input_shape, n_hidden)

    return concarne.patterns.MultiViewPattern(phi=phi, psi=psi, beta=beta, target_var=target_var,
                                              context_var=context_var)


# ########################## Main ###############################
def main(pattern, data_representation, procedure, num_epochs, batchsize):
    print("Pattern: {}".format(pattern))
    print("Data representation: {}".format(data_representation))
    print("Training procedure: {}".format(procedure))
    print("#Epochs: {}".format(num_epochs))
    print("Batchsize: {}".format(batchsize))

    if pattern == "multiview":
        assert (procedure == "simultaneous")

    iterate_context_minibatches = None

    # ------------------------------------------------------
    # Load data and prepare Theano variables
    print("Loading data...")
    X_train, C_train, X_sup, y_sup, X_val, y_val, X_test, y_test, num_classes = load_handwritten_data()

    input_var = T.tensor4('inputs')
    target_var = T.ivector('targets')

    # prepare context data
    if data_representation == 'discrete':

        # discretize context data into 32 classes using kmeans
        kmeans = cluster.KMeans(n_clusters=32, n_init=100)  # init=centers)
        C_train = kmeans.fit_predict(C_train)
        context_var = T.ivector('contexts')

        if pattern in ['direct', 'multiview']:
            # for the direct pattern, we need to explicitly apply one-hot representation to the data
            v = T.vector()
            one_hot = theano.function([v], lasagne.utils.one_hot(v))
            C_train = one_hot(C_train)
            context_var = T.matrix('contexts')
    else:

        # subsample context to have the same dimension as the intermediate representation (required for direct pattern)
        C_train = C_train[:, ::2]
        context_var = T.matrix('contexts')

    # ------------------------------------------------------
    # Build pattern
    learning_rate = 0.003
    momentum = 0.5
    loss_weights = {'target_weight': 0.5, 'context_weight': 0.5}  # default to uniform weighting
    if pattern == "direct":
        pattern = build_direct_pattern(input_var, target_var, context_var, input_shape=(None, 1, 32, 32),
                                       n_hidden=32, num_classes=num_classes)

    elif pattern == "multitask":
        pattern = build_multitask_pattern(input_var, target_var, context_var,
                                          input_shape=(None, 1, 32, 32), n_hidden=32, num_classes=num_classes,
                                          n_out_context=32, discrete=data_representation == 'discrete')

    elif pattern == "multiview":
        pattern = build_multitask_pattern(input_var, target_var, context_var,
                                          input_shape=(None, 1, 32, 32), n_hidden=32, num_classes=num_classes,
                                          n_out_context=32, discrete=data_representation == 'discrete')
        loss_weights = {'target_weight': 0.99, 'context_weight': 0.01}

    else:
        print("Pattern {} not implemented.".format(pattern))
        return

    # ------------------------------------------------------
    # Get the loss expression for training

    trainer = concarne.training.PatternTrainer(pattern,
                                               num_epochs,
                                               learning_rate,
                                               batchsize,
                                               momentum,
                                               procedure,
                                               loss_weights['target_weight'],
                                               loss_weights['context_weight'])
    print("Starting training...")
    print(len(X_train))
    print(len(C_train))
    trainer.fit_XC_XY(X_train, [C_train], X_sup, y_sup, X_val=X_test, y_val=y_test, verbose=True)

    print("=================")
    print("Test score...")
    trainer.score(X_test, y_test, verbose=True)

    return pattern


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("pattern", type=str, help="which pattern to use",
                        default='multitask', nargs='?',
                        choices=['direct', 'multitask', 'multiview'])
    parser.add_argument("data_representation", type=str, help="which context data to load",
                        default='discrete', nargs='?',
                        choices=['continuous', 'discrete'])
    parser.add_argument("training_procedure", type=str, help="which training procedure to use",
                        default='pretrain_finetune', nargs='?',
                        choices=['decoupled', 'pretrain_finetune', 'simultaneous'])
    parser.add_argument("--num_epochs", type=int, help="number of epochs for SGD", default=100, required=False)
    parser.add_argument("--batchsize", type=int, help="batch size for SGD", default=20, required=False)
    args = parser.parse_args()

    pattern = main(args.pattern, args.data_representation, args.training_procedure, args.num_epochs, args.batchsize)
