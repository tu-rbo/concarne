"""

Direct pattern. 

"""
__all__ = [
  "DirectPattern",
]

from .base import Pattern


#import lasagne.objectives

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
