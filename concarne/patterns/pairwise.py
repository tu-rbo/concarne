"""

Pairwise patterns. 

"""
__all__ = [
  "PairwiseTransformationPattern",
  "PairwisePredictTransformationPattern",
]

from .base import Pattern

import theano.tensor as T
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
      x_j ----> s_j -->  ~z
           phi       beta

    Note that self.side_var should represent x_j, whereas self.input_var 
    represents self.x_i. The variable ``z'' in the picture is then represented
    by side_transform_var.
    
    The subclass of this pattern decides what beta looks like.
    

    Parameters
    ----------
    side_transform_var: a Theano variable 
      Variable representing the transformation.
      This will usually be the classification / regression target of beta.
    """
  
    @property
    def default_target_objective(self):
        return lasagne.objectives.categorical_crossentropy  
  
    @property  
    def default_side_objective(self):
        return lasagne.objectives.squared_error
  
    def __init__(self, side_transform_var=None, side_transform_shape=None, **kwargs):
        # we should have side_shape=input_shape
        if 'input_shape' in kwargs:
          if 'side_shape' in kwargs:
            if kwargs['input_shape'] != kwargs['side_shape']:
              raise Exception("side_shape should be omitted - it is required to have"
                              " same value as input_shape in the pairwise patterns!")
        
          kwargs['side_shape'] = kwargs['input_shape']
        
        self.side_transform_shape = side_transform_shape

        self.side_transform_var = side_transform_var
        assert (self.side_transform_var is not None)

        # layer collecting the phi outputs, connecting them and passing them to beta
        self.side_connect_layer = None

        super(PairwiseTransformationPattern, self).__init__(**kwargs)

#         self._create_target_objective()
#         self._create_side_objective()                                     

    def get_side_objective(self, input, target):
        assert (self.input_var is not None)
        assert (self.side_var is not None)
        
        if self.side_loss_fn is None:
            fn = self.default_side_objective
        else:
            #print ("Side loss is function object: %s" % str(self.side_loss_fn))
            fn = self.side_loss_fn
        
        return fn(
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
    z is used as given information about the transformation between pairs
    of input pairs. The function beta is then used to predict z from a pair
    (x_i, x_j)::

                   psi
       x_i ----> s_i ------> y
            phi      \\
                      \\
       x_j ----> s_j ------> ~z
            phi       beta(s_i, s_j)

    Note that self.side_var should represent x_j, whereas self.input_var 
    represents self.x_i. The variable ``z'' in the picture is then represented
    by side_transform_var.
    

    Parameters
    ----------
    beta_input_mode: str
      Determines the mode how ``beta'' gets its input:
      ``diff'' means [phi(x_i),-phi(x_j)], ``stacked'' means [phi(x_i), phi(x_j)]
    """
  
    def __init__(self, beta_input_mode="diff", **kwargs):
        if 'representation_shape' not in kwargs:
            try:
              kwargs['representation_shape'] = kwargs['side_shape']
              print ("WARN: representation_shape not passed; assuming equals"
                     " dimensionality of side information, i.e. side_shape")
            except:
              pass

        #assert (beta_input_mode in ['diff', 'distance','stacked'])
        assert (beta_input_mode in ['diff', 'stacked'])
        self.beta_input_mode = beta_input_mode
        super(PairwisePredictTransformationPattern, self).__init__(**kwargs)


    @property  
    def default_beta_input(self):
        if self.side_connect_layer is None:
          # create connection layer
          if self.beta_input_mode == "diff":
            self.side_connect_layer = lasagne.layers.ElemwiseSumLayer(
                [self.phi, self.phi], name="stack")
          elif self.beta_input_mode == "stacked":
            self.side_connect_layer = lasagne.layers.ConcatLayer(
                [self.phi, self.phi], name="concat")
        return self.side_connect_layer

    @property  
    def default_beta_output_shape(self):
        return self.side_transform_shape
        
    def get_beta_output_for(self, input_i, input_j, **kwargs):        
        phi_i_output = lasagne.layers.get_output(self.phi, input_i, **kwargs)
        phi_j_output = lasagne.layers.get_output(self.phi, input_j, **kwargs)
        
        if self.beta is None:
            if self.beta_input_mode == "diff":
                return phi_i_output-phi_j_output
            #elif self.beta_input_mode == "distance":
            #    return T.sqrt((phi_i_output-phi_j_output)**2
            elif self.beta_input_mode == "stacked":
                return T.concatenate([phi_i_output, phi_j_output], axis=1)
            
        else:
            # beta is a function
            if self.beta_input_mode == "diff":
                beta_in = [phi_i_output, -phi_j_output]
            elif self.beta_input_mode == "stacked":
                beta_in = [phi_i_output, phi_j_output]

            return self.get_output_for_function(self.beta, beta_in, **kwargs)
            
