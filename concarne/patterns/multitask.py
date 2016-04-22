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
        
    def __init__(self, **kwargs):
        super(MultiTaskPattern, self).__init__(**kwargs)

        assert(self.beta is not None)

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
                self.get_beta_output_for(self.input_var), self.side_var
            ).mean()

