"""

Pairwise patterns. 

"""
__all__ = [
  "PairwisePredictTransformationPattern",
]

from .base import Pattern

import lasagne.objectives
import lasagne.layers

import numpy as np

class SiameseLayer(lasagne.layers.Layer):
    def __init__(self, input_var1=None, input_var2=None, concatenation_axis=1, **kwargs):
        super(SiameseLayer, self).__init__(**kwargs)
        self.concatenation_axis = concatenation_axis
        

    def get_output_shape_for(self, input_shapes):
        """
        Computes the output shape of this layer, given a list of input shapes.

        Parameters
        ----------
        input_shape : list of tuple
            A list of tuples, with each tuple representing the shape of one of
            the inputs (in the correct order). These tuples should have as many
            elements as there are input dimensions, and the elements should be
            integers or `None`.

        Returns
        -------
        tuple
            A tuple representing the shape of the output of this layer. The
            tuple has as many elements as there are output dimensions, and the
            elements are all either integers or `None`.

        """
        full_shape = None
        for _is in input_shapes:
            if full_shape is None:
                full_shape = np.zeros( (len(_is),) )
            assert (full_shape[0] == _is[0])
            full_shape [1:] += _is[1:]
        return full_shape

    def get_output_for(self, inputs, **kwargs):
        """
        Propagates the given inputs through this layer (and only this layer).

        Parameters
        ----------
        inputs : list of Theano expressions
            The Theano expressions to propagate through this layer.

        Returns
        -------
        Theano expressions
            The output of this layer given the inputs to this layer.

        Notes
        -----
        This is called by the base :meth:`lasagne.layers.get_output()`
        to propagate data through a network.

        This method should be overridden when implementing a new
        :class:`Layer` class with multiple inputs. By default it raises
        `NotImplementedError`.
        """
        return np.concatenate( inputs, axis=self.concatenation_axis )
        
        
    

class PairwisePredictTransformationPattern(Pattern):
    """
    The :class:`PairwisePredictTransformationPattern` is a contextual pattern where 
    c is used as given information about the transformation between pairs
    of input pairs. The function beta is then used to predict c from a pair
    (x_i, x_j).
    
                   psi
    x_i ----> s_i ------> y
         phi      \
                   \
    x_j ----> s_j ------> ~c
         phi       beta(s_i, s_j)

    Note that self.context_var should represent x_j, whereas self.input_var 
    represents self.x_i. The variable ``c'' in the picture is then represented
    by context_transform_var.
    

    Parameters
    ----------
    context_transform_var: a Theano variable representing the transformation.
    """
  
    def __init__(self, context_transform_var=None, **kwargs):
        super(PairwisePredictTransformationPattern, self).__init__(**kwargs)
        
        self.context_transform_var = context_transform_var
        assert (self.context_transform_var is not None)
        
        if self.target_loss is None:
            assert (self.input_var is not None)
            assert (self.target_var is not None)
            self.target_loss = lasagne.objectives.categorical_crossentropy(
                self.get_psi_output_for(self.input_var), self.target_var
            ).mean()

        if self.context_loss is None:
            assert (self.input_var is not None)
            assert (self.context_var is not None)
            self.sx = self.get_phi_output_for(self.input_var)
            self.sc = self.get_phi_output_for(self.context_var)
            self.context_loss = lasagne.objectives.squared_error(
                #self.get_beta_output_for(self.input_var, self.context_var), 
                self.sx - self.sc,
                self.context_transform_var
            ).mean()

#    def get_beta_output_for(self, input_i, input_j, **kwargs):
#        phi_i_output = self.phi.get_output_for(input_i, **kwargs)
#        phi_j_output = self.phi.get_output_for(input_j, **kwargs)
#        if self.beta is not None:
#            return self.beta.get_output_for([phix_output, phic_output], **kwargs)
#        else:
#            return phi_output
