__all__ = ["Pattern"]

import lasagne.objectives

class Pattern(object):
    """
    The :class:`Pattern` class represents a contextual pattern and
    should be subclassed when implementing a new pattern.

    It is similar to :class:`lasagne.layers.Layer` and mimics some of 
    its functionality, but does not inherit from it.

    Parameters
    ----------
    name : a string or None
        An optional name to attach to this layer.
    """
    def __init__(self, 
                 phi, psi, beta=None, 
                 target_var=None, context_var=None, 
                 target_loss=None, context_loss=None,
                 validation_loss=None,
                 name=None):
        self.phi = phi
        self.psi = psi
        self.beta = beta

        self.input_layer = phi.input_layer
        self.input_var = self.input_layer.input_var
        
        self.target_var = target_var
        self.context_var = context_var

        self.target_loss = target_loss
        self.context_loss = context_loss
        self.validation_loss = validation_loss
    
        self.name = name
            
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

    @property
    def output_shape(self):
        return self.get_output_shape_for(self.input_var)

    def get_params(self, **tags):
        """
        Returns a list of all the Theano variables that parameterize the 
        pattern.

        By default, all parameters that participate in the forward pass will be
        returned. The list can optionally be filtered by
        specifying tags as keyword arguments. For example, ``trainable=True``
        will only return trainable parameters, and ``regularizable=True``
        will only return parameters that can be regularized (e.g., by L2
        decay).

        Parameters
        ----------
        **tags (optional)
            tags can be specified to filter the list. Specifying ``tag1=True``
            will limit the list to parameters that are tagged with ``tag1``.
            Specifying ``tag1=False`` will limit the list to parameters that
            are not tagged with ``tag1``. Commonly used tags are
            ``regularizable`` and ``trainable``.

        Returns
        -------
        list of Theano shared variables
            A list of variables that parameterize the layer

        Notes
        -----
        For layers without any parameters, this will return an empty list.
        """
        params = self.psi.get_params(**tags)
        if self.beta is not None:
            params += self.beta.get_params(**tags)
        params += self.phi.get_params(**tags) 
        return params

    def get_output_shape_for(self, input_shape):
        """
        Computes the output shape of this layer, given an input shape.

        Parameters
        ----------
        input_shape : tuple
            A tuple representing the shape of the input. The tuple should have
            as many elements as there are input dimensions, and the elements
            should be integers or `None`.

        Returns
        -------
        tuple
            A tuple representing the shape of the output of this layer. The
            tuple has as many elements as there are output dimensions, and the
            elements are all either integers or `None`.

        Notes
        -----
        This method will typically be overridden when implementing a new
        :class:`Layer` class. By default it simply returns the input
        shape. This means that a layer that does not modify the shape
        (e.g. because it applies an elementwise operation) does not need
        to override this method.
        """
        phi_output_shape = self.phi.get_output_shape_for(input_shape)
        return self.psi.get_output_shape_for(phi_output_shape)

    def get_output(self, **kwargs):
        return self.get_output_for(self.input_var, **kwargs)

    def get_output_for(self, input, **kwargs):
        return self.get_psi_output_for(input, **kwargs)
        
    def get_psi_output_for(self, input, **kwargs):
        phi_output = self.phi.get_output_for(input, **kwargs)
        return self.psi.get_output_for(phi_output, **kwargs)

    def get_phi_output_for(self, input, **kwargs):
        return self.phi.get_output_for(input, **kwargs)

    def training_loss(self, target_weight=0.5, context_weight=0.5):
        return target_weight * self.target_loss \
            + context_weight * self.context_loss