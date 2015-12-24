"""

Multi-view pattern. 

"""
__all__ = [
  "MultiViewPattern",
]

from .base import Pattern

import lasagne.objectives

class MultiViewPattern(Pattern):
    """
    The :class:`MultiViewPattern` is uses additional input (modality) c for
    learning beta(c) -> s', such that s' ~ s, with s=phi(x)::
    
                psi
      x ----> s -----> y
        phi  /  
            / beta(c)
           /
      c ---
       
    """ 
  
    @property
    def default_target_objective(self):
        return lasagne.objectives.categorical_crossentropy  
  
    @property  
    def default_context_objective(self):
        return lasagne.objectives.squared_error

    @property  
    def default_phi_input(self):
        if self.input_layer is None:
            # create input layer
            print ("Creating input layer for phi")
            input_dim = self.input_shape
            if isinstance(self.input_shape, int):
                input_dim = (None, self.input_shape)
            self.input_layer = lasagne.layers.InputLayer(shape=input_dim,
                                        input_var=self.input_var)

        return self.input_layer
        
    @property  
    def default_beta_input(self):
        if self.input_layer is None:
            # create input layer
            print ("Creating input layer for beta")
            input_dim = self.input_shape
            if isinstance(self.input_shape, int):
                input_dim = (None, self.input_shape)
            self.input_layer = lasagne.layers.InputLayer(shape=input_dim,
                                        input_var=self.input_var)

        return self.input_layer
        
    def __init__(self, **kwargs):
        self.context_input_layer = None
        super(MultiViewPattern, self).__init__(**kwargs)

        assert(self.beta is not None)

        self._create_target_objective()
        self._create_context_objective()                                     

    def _create_context_objective(self):
        if self.context_loss is None:
            assert (self.input_var is not None)
            assert (self.context_var is not None)
            
            if self.context_loss_fn is None:
                fn = self.default_context_objective
            else:
                #print ("Context loss is function object: %s" % str(self.context_loss_fn))
                fn = self.context_loss_fn
            
            self.context_loss = fn(
                self.get_beta_output_for(self.context_var), 
                self.get_phi_output_for(self.input_var)
            ).mean()
