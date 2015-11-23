"""

Direct pattern. 


Provides

"""
__all__ = [
  "DirectPattern",
]

from .base import Pattern


#import lasagne.objectives

class DirectPattern(Pattern):
    """
    The :class:`DirectPattern` is the simplest contextual pattern where 
    s is used directly as the target representation for x:
    x --> s --> y
          |
          c


    Parameters
    ----------
    incoming : a :class:`Layer` instance or a tuple
        The layer feeding into this layer, or the expected input shape.
    name : a string or None
        An optional name to attach to this layer.
    """
  
    def __init__(self, **kwargs):
        assert('beta' not in kwargs)
        super(DirectPattern, self).__init__(**kwargs)
