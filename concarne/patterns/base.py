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
    phi : a lasagne layer for computing the intermediate representation 
        s = phi(x) from the input x
    psi : a lasagne layer for computing the prediction of the target
        from the intermediate representation s, psi(s)=y
    target_var : Theano variable representing the target
        Required for formulating the target loss.
    context_var: Theano variable representing the target
        The semantics of this variable depend on the pattern.
        Note that additional context variables might be required by a pattern.
    target_loss: Theano expression for the optimizing the target (optional).
        All patterns have standard objectives applicable here
    context_loss: Theano expression for the contextual loss (optional).
        All patterns have standard objectives applicable here
    name : a string or None
        An optional name to attach to this layer.
    """
    def __init__(self, 
                 phi, psi, beta=None, 
                 target_var=None, context_var=None, 
                 target_loss=None, context_loss=None,
                 name=None):
        self.phi = phi
        self.psi = psi
        self.beta = beta

        self.input_layer = lasagne.layers.get_all_layers(phi)[0]
        self.input_var = self.input_layer.input_var
        
        self.target_var = target_var
        self.context_var = context_var

        self.target_loss = target_loss
        self.context_loss = context_loss
    
        self.name = name

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

        # check between tags that belong to the pattern and those that belong to the layers
        pattern_keys = ['phi', 'psi', 'beta']
        pattern_tags = dict()
        layer_tags = dict()
        for key in tags:
            if key in pattern_keys:
                pattern_tags[key] = tags[key]
            else:
                layer_tags[key] = tags[key]

        result = ['phi', 'psi', 'beta']

        only = set(tag for tag, value in pattern_tags.items() if value)
        if only:
            if len(only) > 1:
                print('WARNING: more than one positive tag results in an empty parameter set. tags: {}'.format(pattern_tags))
            # retain all parameters that have all of the tags in `only`
            result = [param for param in result
                      if not (only - set([param]))]

        exclude = set(tag for tag, value in tags.items() if not value)
        if exclude:
            # retain all parameters that have none of the tags in `exclude`
            result = [param for param in result
                      if not (set([param]) & exclude)]

        # get the parameters for the functions that fit the pattern tags
        params = []
        for param, network in [('phi', self.phi), ('psi', self.psi), ('beta', self.beta)]:
            if param in result and network is not None:
                params += lasagne.layers.get_all_params(network, **layer_tags)

        if len(params) == 0:
            print('WARNING: empty parameter set. tags: {}'.format(pattern_tags))

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

    def get_output_for(self, input=None, **kwargs):
        if input is None:
            input = self.input_var
        return self.get_psi_output_for(input, **kwargs)
        
    def get_psi_output_for(self, input=None, **kwargs):
        if input is None:
            input = self.input_var
        return lasagne.layers.get_output(self.psi, inputs=input)

    def get_beta_output_for(self, input=None, **kwargs):
        if input is None:
            input = self.input_var
        return lasagne.layers.get_output(self.beta, inputs=input)

    def get_phi_output_for(self, input=None, **kwargs):
        if input is None:
            input = self.input_var
        return lasagne.layers.get_output(self.phi, inputs=input)

    def training_loss(self, target_weight=0.5, context_weight=0.5):
        return (target_weight * self.target_loss
            + context_weight * self.context_loss)