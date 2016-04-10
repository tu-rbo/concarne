"""

Pairwise patterns. 

"""
__all__ = [
  "PairwiseTransformationPattern",
  "PairwisePredictTransformationPattern",
]

from .base import Pattern

import lasagne.objectives
import lasagne.layers

class PairwiseTransformationPattern(Pattern):
    """
    Base class for all pairwise transformation patterns. The general framework
    is as follows::
    
                   psi
      x_i ----> s_i ------> y
           phi      \\
                     \\
      x_j ----> s_j -->  ~c
           phi       beta

    Note that self.side_var should represent x_j, whereas self.input_var 
    represents self.x_i. The variable ``c'' in the picture is then represented
    by side_transform_var.
    
    The subclass of this pattern decides what beta looks like.
    

    Parameters
    ----------
    side_transform_var: a Theano variable representing the transformation.
    """
  
    @property
    def default_target_objective(self):
        return lasagne.objectives.categorical_crossentropy  
  
    @property  
    def default_side_objective(self):
        return lasagne.objectives.squared_error
  
    def __init__(self, side_transform_var=None, side_transform_shape=None, **kwargs):
        # we should have side_shape=input_shape
        if 'side_shape' in kwargs:
          if 'input_shape' in kwargs:
            if kwargs['input_shape'] != kwargs['side_shape']:
              raise Exception("side_shape should be omitted - it is required to have"
                              " same value as input_shape in the pairwise patterns!")
            kwargs['side_shape'] = kwargs['input_shape']
        
        self.side_input_layer = None
        self.side_transform_shape = side_transform_shape

        self.side_transform_var = side_transform_var
        assert (self.side_transform_var is not None)

        super(PairwiseTransformationPattern, self).__init__(**kwargs)

        self._create_target_objective()
        self._create_side_objective()                                     

    def _create_side_objective(self):
        if self.side_loss is None:
            assert (self.input_var is not None)
            assert (self.side_var is not None)
            
            if self.side_loss_fn is None:
                fn = self.default_side_objective
            else:
                #print ("Side loss is function object: %s" % str(self.side_loss_fn))
                fn = self.side_loss_fn
            
            self.side_loss = fn(
                self.get_beta_output_for(self.input_var, self.side_var), 
                self.side_transform_var
            ).mean()


    def get_beta_output_for(self, input_i, input_j, **kwargs):
        raise NotImplementedError()

    @property
    def training_input_vars(self):
        return (self.input_var, self.target_var, self.side_var, self.side_transform_var)
          
    @property
    def side_vars(self):
        return (self.side_var, self.side_transform_var)        

class PairwisePredictTransformationPattern(PairwiseTransformationPattern):
    """
    The :class:`PairwisePredictTransformationPattern` is a pattern where 
    c is used as given information about the transformation between pairs
    of input pairs. The function beta is then used to predict c from a pair
    (x_i, x_j)::

                   psi
       x_i ----> s_i ------> y
            phi      \\
                      \\
       x_j ----> s_j ------> ~c
            phi       beta(s_i, s_j)

    Note that self.side_var should represent x_j, whereas self.input_var 
    represents self.x_i. The variable ``c'' in the picture is then represented
    by side_transform_var.
    

    Parameters
    ----------
    side_transform_var: a Theano variable representing the transformation.
    """
  
    def __init__(self, **kwargs):
        if 'representation_shape' not in kwargs:
            try:
              kwargs['representation_shape'] = kwargs['side_shape']
              print ("WARN: representation_shape not passed; assuming equals"
                     " dimensionality of side information, i.e. side_shape")
            except:
              pass
        super(PairwisePredictTransformationPattern, self).__init__(**kwargs)

    @property  
    def default_beta_input(self):
        if self.side_input_layer is None:
            # create input layer
            #print ("Creating input layer for beta")
            side_dim = self.representation_shape
            if isinstance(side_dim, int):
                side_dim = (None, side_dim)
            self.side_input_layer = lasagne.layers.InputLayer(shape=side_dim,
                                        input_var=self.side_var)
        return self.side_input_layer

    @property  
    def default_beta_output_shape(self):
        return self.side_transform_shape

    def get_beta_output_for(self, input_i, input_j, **kwargs):
        phi_i_output = self.phi.get_output_for(input_i, **kwargs)
        phi_j_output = self.phi.get_output_for(input_j, **kwargs)
        diff = phi_i_output-phi_j_output
        if self.beta is not None:
            return self.beta.get_output_for(diff, **kwargs)
        else:
            return diff