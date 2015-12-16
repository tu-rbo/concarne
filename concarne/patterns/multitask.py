"""

Multi-task pattern. 

"""
__all__ = [
  "MultiTaskPattern",
]

from .base import Pattern

import lasagne.objectives

class MultiTaskPattern(Pattern):
    """
    The :class:`MultiTaskPattern` uses additional labels c for
    learning beta(phi(x)) -> c:
    
                psi
    x ----> s -----> y
      phi    \  
              \ beta(s)
               \
                ---> c
    """ 
  
    @property
    def default_target_objective(self):
        return lasagne.objectives.categorical_crossentropy  
  
    @property  
    def default_context_objective(self):
        return lasagne.objectives.squared_error

    def __init__(self, **kwargs):
        super(MultiTaskPattern, self).__init__(**kwargs)

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
                self.get_beta_output_for(self.input_var), self.context_var
            ).mean()
            
    def get_beta_output_for(self, input, **kwargs):
        phi_output = self.phi.get_output_for(input, **kwargs)
        return self.beta.get_output_for(phi_output, **kwargs)
