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
        
    def __init__(self, **kwargs):
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
            
    def get_beta_output_for(self, input, **kwargs):
        return self.beta.get_output_for(input, **kwargs)
