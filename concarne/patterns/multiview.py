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
    The :class:`MultiViewPattern` is uses additional input (modality) s for
    learning beta(s) -> z', such that z' ~ z, with z=phi(x)::
    
                psi
      x ----> z -----> y
        phi  /  
            / beta(s)
           /
      s ---
       
    """ 
  
    @property
    def default_target_objective(self):
        return lasagne.objectives.categorical_crossentropy  
  
    @property  
    def default_side_objective(self):
        return lasagne.objectives.squared_error
        
    @property  
    def default_beta_input(self):
        if self.side_input_layer is None:
            # create input layer
            #print ("Creating input layer for beta")
            side_dim = self.side_shape
            if isinstance(self.side_shape, int):
                side_dim = (None, self.side_shape)
            self.side_input_layer = lasagne.layers.InputLayer(shape=side_dim,
                                        input_var=self.side_var)

        return self.side_input_layer
        
    @property  
    def default_beta_output_shape(self):
        return self.representation_shape
                
    def __init__(self, **kwargs):
        self.side_input_layer = None
        super(MultiViewPattern, self).__init__(**kwargs)

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
                self.get_beta_output_for(self.side_var), 
                self.get_phi_output_for(self.input_var)
            ).mean()
