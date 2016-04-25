import lasagne
from lasagne import nonlinearities
from lasagne import init

import theano
import theano.tensor as T

import concarne
import concarne.lasagne

import numpy as np


class TestMultivariateDenseBase(object):

    def setup(self):
        self.X = np.array([
                [1, 2, 10],
                [1, 2, 20],
                [2, 2, 30],
                [2, 2, 10],
            ])
        self.Y = np.array([
                [0,0],
                [0,1],
                [1,2],
                [1,0],
            ], dtype=np.int32)

        self.input_var = T.matrix('inputs')
        self.target_var = T.imatrix('targets')

        self.univariate_target_var = T.ivector('targets_uni')

        self.l_in = lasagne.layers.InputLayer(shape=(None, 3),
                                         input_var=self.input_var)
                                     
      
        self.network = concarne.lasagne.MultivariateDenseLayer(self.l_in, [2,3])

    def test_dense_vs_multivariate_dense(self):
        # build normal dense layer
        self.target_var2 = T.ivector('targets2')
        l2 = lasagne.layers.DenseLayer(self.l_in, 2, nonlinearity=lasagne.nonlinearities.softmax)

        res2 = l2.get_output_for(self.X)
#          print res2
#          print res2.shape

#         loss2 = lasagne.objectives.categorical_crossentropy(
#             lasagne.layers.get_output(l2, self.input_var),
#             self.target_var)
#         loss2_fn = theano.function([self.input_var, self.target_var], loss2)
#         print (loss2_fn(self.X, self.Y))        
    
        # now compare if weights are set equally
        l = self.network
        l.W0.set_value(l2.W.get_value())
        l.b0.set_value(l2.b.get_value())

        res = l.get_output_for(self.X)
        
        assert (np.all(res[0].eval() == res2.eval()))

    def test_multivariate_dense_shape(self):
        l = self.network

        res = l.get_output_for(self.X)
#         print ([r.eval() for r in res])
#         print ([r.eval().shape for r in res])

        res_shape = l.get_output_shape_for(self.X.shape)
        #print res_shape

        assert (res_shape[0] == (4,2) )
        assert (res_shape[1] == (4,3) )

    def test_multivariate_categorical_crossentropy(self):
        l = self.network
        
        prediction = lasagne.layers.get_output(l, self.input_var, deterministic=True)
        argmax_prediction = [T.argmax(p, axis=1) for p in prediction]
    
        loss = concarne.lasagne.multivariate_categorical_crossentropy(
            prediction,
            self.target_var)
        
        loss_fn = theano.function([self.input_var, self.target_var], loss)
        #print (loss_fn(self.X, self.Y))
    
        #print l.W0.eval()
        #print l.W1.eval()
        # random weights that are known to work as initialization
        l.W0.set_value(
            [[ 0.09172429, -1.05014537],
             [-0.22989627,  0.81534704],
             [-0.19074303, -0.12567531]])
        l.W1.set_value(
            [[-0.01287729,  0.7374191,   0.89021668],
             [-0.51809152, -0.74111272,  0.30311176],
             [ 0.37741316, -0.26230205, -0.30029486]])
    
        #######
        params = lasagne.layers.get_all_params(l, trainable=True)
        updates = lasagne.updates.nesterov_momentum(
                loss.mean(), params, learning_rate=0.01, momentum=0.9)
        train_fn = theano.function([self.input_var, self.target_var], [loss] + argmax_prediction, updates=updates)
    
        for i in range(200):
            #print ("epoch %d" % i)
            res = train_fn(self.X, self.Y)
            #print (res)

#         print np.vstack([res[-2], res[-1]])
#         print ( self.Y.T)
        #assert (np.all(np.vstack([res[-2], res[-1]]) == self.Y.T))
    
        err_fn = theano.function([self.input_var, self.target_var],
              T.mean([ T.eq(T.argmax(p, axis=1), self.target_var[:,i])
                       for (i, p) in enumerate(prediction)], dtype=theano.config.floatX))
        assert(err_fn(self.X, self.Y) == 1.)
   
        
if __name__ == "__main__":
    t = TestMultivariateDenseBase()

    t.setup()
    t.test_dense_vs_multivariate_dense()
    
    t.setup()
    t.test_multivariate_dense_shape()

    t.setup()
    t.test_multivariate_categorical_crossentropy()
    