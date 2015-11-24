"""

Direct pattern. 

"""
__all__ = [
  "DirectPattern",
]

from .base import Pattern

import lasagne.objectives

class DirectPattern(Pattern):
    """
    The :class:`DirectPattern` is the simplest contextual pattern where 
    c is used directly as the target representation s = phi(x):
    x ----> s -----> y
      phi   |  psi
            c  
    """
  
    def __init__(self, **kwargs):
        assert('beta' not in kwargs)
        super(DirectPattern, self).__init__(**kwargs)

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
                self.get_phi_output_for(self.input_var), self.context_var
            ).mean()