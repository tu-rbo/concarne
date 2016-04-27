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
    The :class:`MultiTaskPattern` uses additional labels z for
    learning beta(phi(x)) -> z::
    
                psi
      x ----> s -----> y
        phi    \\  
                \\ beta(s)
                 \\
                ---> z
    """
  
    @property
    def default_target_objective(self):
        return lasagne.objectives.categorical_crossentropy  
  
    @property  
    def default_side_objective(self):
        return lasagne.objectives.squared_error

    @property  
    def default_beta_input(self):
        return self.phi

    @property  
    def default_beta_output_shape(self):
        return self.side_shape

    @property
    def validation_side_input_vars(self):
        return (self.input_var, )
        
    @property
    def validation_side_target_var(self):
        return self.side_var
        
    def __init__(self, **kwargs):
        assert('beta' in kwargs and kwargs['beta'] is not None)
        super(MultiTaskPattern, self).__init__(**kwargs)

#         self._create_target_objective()
#         self._create_side_objective()                                     

    def get_side_objective(self, input, target):
        if self.side_loss_fn is None:
            fn = self.default_side_objective
        else:
            #print ("Side loss is function object: %s" % str(self.side_loss_fn))
            fn = self.side_loss_fn
        
        return fn(
            self.get_beta_output_for(input), target
        ).mean()
