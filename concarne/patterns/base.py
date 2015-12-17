__all__ = ["Pattern"]

import lasagne.objectives

from ..utils import isfunction


class Pattern(object):
    """
    The :class:`Pattern` class represents a contextual pattern and
    should be subclassed when implementing a new pattern.

    It is similar to :class:`lasagne.layers.Layer` and mimics some of 
    its functionality, but does not inherit from it.

    Parameters
    ----------
    phi : lasagne layer
        a lasagne layer for computing the intermediate representation 
        :math:`\phi(s)=y` from the input x
    psi : lasagne layer
        a lasagne layer for computing the prediction of the target
        from the intermediate representation s, :math:`\psi(s)=y`
    target_var : theano tensor variable
        Theano variable representing the target. Required for formulating the target loss.
    context_var: theano tensor variable
        Theano variable representing the target.
        The semantics of this variable depend on the pattern.
        Note that additional context variables might be required by a pattern.
    target_loss: theano tensor variable, optional
        Theano expression or lasagne objective for the optimizing the 
        target.
        All patterns have standard objectives applicable here
    context_loss: theano tensor variable, optional
        Theano expression or lasagne objective for the contextual 
        loss.
        All patterns have standard objectives applicable here
    name : string, optional
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
        self.target_loss_fn = None
        self.context_loss = context_loss
        self.context_loss_fn = None
    
        self.name = name
        
        if isfunction(self.target_loss):
            self.target_loss_fn = self.target_loss
            self.target_loss = None
        if isfunction(self.context_loss):
            self.context_loss_fn = self.context_loss
            self.context_loss = None

        # tag the parameters of each function with the name of the function
        for fun, fun_name in zip([self.phi, self.psi, self.beta], ['phi', 'psi', 'beta']):
            self._tag_function_parameters(fun, fun_name)
        

    def _tag_function_parameters(self, fun, fun_name):
        """
        Helper function to add the tag `fun_name` (encoding the function name,
        e.g. phi or psi) to the function `fun`
        """
        for l in lasagne.layers.get_all_layers(fun):
            params = l.get_params()
            for p in params:
                if fun_name != 'phi' and 'phi' in l.params[p]:
                    #print ("omitting phi for %s" % str(p))
                    continue
                #print ("adding %s to param %s" % (fun_name, str(p)))
                l.params[p].add(fun_name)
        
    @property
    def training_input_vars(self):
        """Return the theano variables that are required for training.
            
           Usually this will correspond to 
           (input_var, target_var, context_var)
           which is also the default.
            
           Order matters!
            
           Returns
           -------
           tuple of theano tensor variables
        """
        return (self.input_var, self.target_var, self.context_var)
          
    @property
    def context_vars(self):
        """Return the theano variables that are required for training.
            
           Usually this will correspond to 
           (input_var, target_var, context_var)
           which is also the default.
            
           Order matters!

           Returns
           -------
           tuple of theano tensor variables
        """
        return (self.context_var, )
          
            
    @property
    def default_target_objective(self):
        """ Return the default target objective used by this pattern.
            
            The target objective can be overridden by passing the 
            target_loss argument to the constructor of a pattern

            Returns
            -------
            theano expression
        """
        raise NotImplemented()
  
  
    @property  
    def default_context_objective(self):
        """ Return the default contextual objective used by this pattern.
            
            The contextual objective can be overridden by passing the 
            context_loss argument to the constructor of a pattern

            Returns
            -------
            theano expression
        """
        raise NotImplemented()
  
    def _create_target_objective(self, output=None, target=None):
        """
            Helper function that should be called by constructor to build
            the member variable target_loss.
            
            Should be called by the constructor
        """
        
        if output is None:
            output = self.get_psi_output_for(self.input_var)
        if target is None:
            target = self.target_var            
        
        if self.target_loss is None:
            assert (self.input_var is not None)
            assert (self.target_var is not None)
            
            if self.target_loss_fn is None:
                fn = self.default_target_objective
            else:
                #print ("Target loss is function object: %s" % str(self.target_loss_fn))
                fn = self.target_loss_fn
            
            # define target loss
            self.target_loss = fn(output, target).mean()

                

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
        For patterns without any parameters, this will return an empty list.
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
        if target_weight == 0.:
            return context_weight * self.context_loss
        elif context_weight == 0.:
            return target_weight * self.target_loss
        else:
            return target_weight * self.target_loss \
                + context_weight * self.context_loss