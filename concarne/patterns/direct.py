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
    The :class:`DirectPattern` is the simplest learning with side information pattern   
    where c is used directly as the target representation s = phi(x)::
    
     x ----> s -----> y
       phi   |  psi
             c  

    """
  
    @property
    def default_target_objective(self):
        return lasagne.objectives.categorical_crossentropy  
  
    @property  
    def default_side_objective(self):
        return lasagne.objectives.squared_error

  
    def __init__(self, **kwargs):
        assert('beta' not in kwargs)
        super(DirectPattern, self).__init__(**kwargs)

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
                self.get_phi_output_for(self.input_var), self.side_var
            ).mean()
            
            