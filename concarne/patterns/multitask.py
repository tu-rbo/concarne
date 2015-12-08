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

    # TODO: parameters kwargs are hidden ... :(
    def __init__(self, **kwargs):
        super(MultiTaskPattern, self).__init__(**kwargs)

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
                self.get_beta_output_for(self.input_var), self.context_var
            ).mean()
            
    def get_beta_output_for(self, input, **kwargs):
        phi_output = self.phi.get_output_for(input, **kwargs)
        return self.beta.get_output_for(phi_output, **kwargs)
