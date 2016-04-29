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
    where side information z is used directly as the target representation s = phi(x)::
    
     x ----> s -----> y
       phi   |  psi
             z  

    """
  
    @property
    def default_target_objective(self):
        return lasagne.objectives.categorical_crossentropy  
  
    @property  
    def default_side_objective(self):
        return lasagne.objectives.squared_error

    @property
    def side_target_var(self):
        return self.side_var

    def get_beta_output_for(self, input=None, **kwargs):
        return self.get_phi_output_for(input=input, **kwargs)
  
    def __init__(self, **kwargs):
        assert('beta' not in kwargs)
        super(DirectPattern, self).__init__(**kwargs)

#         self._create_target_objective()
#         self._create_side_objective()                                     

    def get_side_objective(self, input, target):
        if self.side_loss_fn is None:
            fn = self.default_side_objective
        else:
            #print ("Side loss is function object: %s" % str(self.side_loss_fn))
            fn = self.side_loss_fn
        
        return fn(
            self.get_phi_output_for(input), target
        ).mean()
