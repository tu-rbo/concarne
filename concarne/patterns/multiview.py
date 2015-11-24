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
    learning beta(c) -> s', such that s' ~ s, with s=phi(x):
    
                psi
    x ----> s -----> y
      phi  /  
          / beta(c)
         /
    c ---
       
    """ 
  
    def __init__(self, **kwargs):
        super(MultiViewPattern, self).__init__(**kwargs)

        assert(self.beta is not None)

        if self.target_loss is None:
            assert (self.input_var is not None)
            assert (self.target_var is not None)
            self.target_loss = lasagne.objectives.categorical_crossentropy(
                self.get_psi_output_for(self.input_var), self.target_var
            ).mean()

        if self.context_loss is None:
            assert (self.input_var is not None)
            assert (self.context_var is not None)
            self.context_loss = lasagne.objectives.squared_error(
                self.get_beta_output_for(self.context_var), 
                self.get_phi_output_for(self.input_var)
            ).mean()
            
    def get_beta_output_for(self, input, **kwargs):
        return self.beta.get_output_for(input, **kwargs)
