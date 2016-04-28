__all__ = ["Pattern"]

from ..utils import isfunction

import lasagne.objectives
from lasagne.layers import get_all_layers
from lasagne.layers import InputLayer
from lasagne.layers import Layer

import numpy as np

import itertools
from collections import OrderedDict
import inspect

import copy
import os
import tarfile

class Pattern(object):
    """
    The :class:`Pattern` class represents a side information pattern and
    should be subclassed when implementing a new pattern.

    It is similar to :class:`lasagne.layers.Layer` and mimics some of 
    its functionality, but does not inherit from it.
    
    === How to implement your own pattern? ===
    The minimal example should implement the following functions:

    - get_side_objective
    - default_target_objective
    - default_side_objective
    - default_beta_input
    - default_beta_output_shape

    Depending on the pattern, you might need to override (usually if you need 
    additional side variables):
    - training_input_vars
    - side_vars 

    If you beta has multiple inputs, you will need to implement:
    - get_beta_output_for

    Optionally, you might want to implement in order to use :method:`Pattern.score_side`:
    - validation_side_input_vars(self):
    - validation_side_target_var(self)

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
    side_var: theano tensor variable
        Theano variable representing the side information.
        The semantics of this variable depend on the pattern.
        Note that additional side variables might be required by a pattern.
    input_shape : int or tuple
        Shape of the input variable
    target_shape : int or tuple
        Shape of the target variable
    side_shape : int or tuple
        Shape of the side information variable
    representation_shape : int or tuple
        Shape of the intermediate representation to be learned
        (for some patterns that may coincide with the side_shape)
    target_loss: theano tensor variable, optional
        Theano expression or lasagne objective for the optimizing the 
        target.
        All patterns have standard objectives applicable here
    side_loss: theano tensor variable, optional
        Theano expression or lasagne objective for the side loss.
        All patterns have standard objectives applicable here
    name : string, optional
        An optional name to attach to this layer.
    """

    PHI_OUTPUT_SHAPE='PHI_OUTPUT_SHAPE'    
    PSI_OUTPUT_SHAPE='PSI_OUTPUT_SHAPE'    
    BETA_OUTPUT_SHAPE='BETA_OUTPUT_SHAPE'    
    
    def __init__(self, 
                 phi, psi, beta=None, 
                 input_var=None, target_var=None, side_var=None, 
                 input_shape=None, target_shape=None, side_shape=None, 
                 representation_shape=None,
                 target_loss=None, side_loss=None,
                 name=None):
        self.phi = phi
        self.psi = psi
        self.beta = beta

        self.input_var = input_var
        self.target_var = target_var
        self.side_var = side_var

        self.input_shape = input_shape
        self.target_shape = target_shape
        self.side_shape = side_shape
        self.representation_shape = representation_shape

        self.target_loss = target_loss
        self.target_loss_fn = None
        self.side_loss = side_loss
        self.side_loss_fn = None
    
        self.name = name
        
        self.input_layer = None
        
        if isfunction(self.target_loss):
            self.target_loss_fn = self.target_loss
            self.target_loss = None
        if isfunction(self.side_loss):
            self.side_loss_fn = self.side_loss
            self.side_loss = None
            
        # convert phi, psi and beta to real lasagne layers if they
        # are passed as a list/dictionary
        if isinstance(phi, list) or isinstance(phi, tuple):
            # if no input layer in list -> build it
            assert (input_var is not None)
            phi = copy.deepcopy(phi)
            self.phi = \
                self._initialize_function('phi', phi, self.default_phi_input,
                                          self.PHI_OUTPUT_SHAPE,
                                          self.representation_shape)
            self.input_layer = lasagne.layers.get_all_layers(self.phi)[0]
        else:
            # extract input layer and variable from the given phi
            self.input_layer = lasagne.layers.get_all_layers(self.phi)[0]
            self.input_var = self.input_layer.input_var
        self.phi._fun_name = "phi"

        if isinstance(psi, list) or isinstance(psi, tuple):
            # if no input layer in list -> build it
            psi = copy.deepcopy(psi)
            self.psi = \
                self._initialize_function('psi', psi, self.default_psi_input,
                                          self.PSI_OUTPUT_SHAPE,
                                          self.target_shape)
            self.psi._fun_name = "psi"
        
        if beta is not None and isinstance(beta, list) or isinstance(beta, tuple):
            # if no input layer in list -> build it
            beta = copy.deepcopy(beta)
            try:
              self.beta = \
                  self._initialize_function('beta', beta, self.default_beta_input,
                                            self.BETA_OUTPUT_SHAPE, 
                                            self.default_beta_output_shape
                                            )
              self.beta._fun_name = "beta"
            except ValueError, e:
              raise Exception("Could not replace BETA_OUTPUT_SHAPE marker --"
                     " is the value returned by self.default_beta_output_shape"
                     " valid? (not None)\n"
                     " Futher hints: " + str(e))
              

        # tag the parameters of each function with the name of the function
        for fun, fun_name in zip([self.phi, self.psi, self.beta], ['phi', 'psi', 'beta']):
            self._tag_function_parameters(fun, fun_name)
        
        self._create_target_objective()
        self._create_side_objective()                                     
        
    def get_side_objective(self, input, target):
        """
        Pattern-specific function to get the theano expression of the side objective.
        
        Must be implemented by each pattern.
        """
        raise NotImplementedError()       
                
    @property
    def training_input_vars(self):
        """Return the theano variables that are required for training.
            
           Usually this will correspond to 
           (input_var, target_var, side_var)
           which is also the default.
            
           Order matters!
            
           Returns
           -------
           tuple of theano tensor variables
        """
        return (self.input_var, self.target_var, self.side_var)

    @property
    def validation_side_input_vars(self):
        """Return the theano input variables for validating the side loss.

           Order matters!
            
           This strongly depends on the pattern and is therefore implemented
           individually. 
           
           It is not obligatory, but you will not be able to compute the side validation
           loss.

           Also see :method:`Pattern.validation_target_var`
            
           Returns
           -------
           tuple of theano tensor variables
        """
        raise NotImplementedError()

    @property
    def validation_target_var(self):
        """Return the theano target variable required for validating
           the side information (optional).

           This returns None per default. Override it the side loss of the pattern is a not a 
           supervised loss. 
            
           This strongly depends on the pattern and is therefore implemented
           individually. 
           
           It is not obligatory, but you will not be able to compute the side validation
           loss during training.

           Also see :method:`Pattern.validation_input_vars`
            
           Returns
           -------
           theano tensor variable
        """
        return None

                    
    @property
    def side_vars(self):
        """Return the theano variables that are required for training.
            
           Usually this will correspond to 
           (input_var, target_var, side_var)
           which is also the default.
            
           Order matters!

           Returns
           -------
           tuple of theano tensor variables
        """
        return (self.side_var, )
          
            
    @property
    def default_target_objective(self):
        """ Return the default target objective used by this pattern.
            (implementation required)
            
            The target objective can be overridden by passing the 
            target_loss argument to the constructor of a pattern

            Returns
            -------
            theano expression
        """
        raise NotImplementedError()
  
  
    @property  
    def default_side_objective(self):
        """ Return the default side objective used by this pattern.
            (implementation required)
            
            The side objective can be overridden by passing the 
            side_loss argument to the constructor of a pattern

            Returns
            -------
            theano expression
        """
        raise NotImplementedError()
  
  
    @property  
    def default_phi_input(self):
        """ Specifies the default input to the function :math:`\phi` in this pattern
            (implementation required)
            
            This may either return a tuple of lasagne layer class and a 
            dictionary containing the params for instantiation of a layer,
            or it contains a lasagne layer object
        
            Per default, this will create/return an input_layer with self.input_var
            of dimensionality self.input_shape
            -------
            Returns:            
            tuple of lasagne layer class and dictionary, or lasagne layer instance
                
        """
        if self.input_layer is None:
            # create input layer
            #print ("Creating input layer for phi")
            input_dim = self.input_shape
            if isinstance(self.input_shape, int):
                input_dim = (None, self.input_shape)
            self.input_layer = lasagne.layers.InputLayer(shape=input_dim,
                                        input_var=self.input_var, name="input")

        return self.input_layer

    @property  
    def default_psi_input(self):
        """ Specifies the default input to the function :math:`\psi` in this pattern

            This may either return a tuple of lasagne layer class and a 
            dictionary containing the params for instantiation of a layer,
            or it contains a lasagne layer object
        
            Per default, this will return the output of :math:`\phi`.
        
            -------
            Returns:            
            tuple of lasagne layer class and dictionary, or lasagne layer instance
        """
        return self.phi

    @property  
    def default_beta_input(self):
        """ Specifies the default input to the function :math:`\beta` in this pattern

            This may either return a tuple of lasagne layer class and a 
            dictionary containing the params for instantiation of a layer,
            or it contains a lasagne layer object
        
            -------
            Returns:            
            tuple of lasagne layer class and dictionary, or lasagne layer instance
        """
        raise NotImplementedError()
        

    @property  
    def default_beta_output_shape(self):
        """Every pattern that uses an auxiliary function beta should
        implement this method which computes the shape.
        
        This is helpful for automatically building beta in nolearn style
        function parameterization
        
        --------
        Returns:       
        int or tuple of ints
        """
        raise NotImplementedError()
                                

    def _get_all_function_layers(self, fun):
        """
         Get only the layers that belong to a certain function
        """
        layers = []
        for l in lasagne.layers.get_all_layers(fun):
          if l._pattern_function == fun._fun_name:
            layers.append(l)
        return layers            
            
    def _tag_function_parameters(self, fun, fun_name):
        """
        Helper function to add the tag `fun_name` (encoding the function name,
        e.g. phi or psi) to the function `fun`
        """
        for l in lasagne.layers.get_all_layers(fun):
            params = l.get_params()
            for p in params:
                if fun_name != 'phi' and 'phi' in l.params[p]:
#                    print ("omitting phi for %s" % str(p))
                    continue
#                print ("adding %s to param %s" % (fun_name, str(p)))
                l.params[p].add(fun_name)
#                print ("  tags: " + str(l.params[p]))



    def _get_params_for(self, name):
        """This method has been adapted from the NeuralFit class in nolearn.
        https://github.com/dnouri/nolearn/blob/master/nolearn/lasagne/base.py
        Copyright (c) 2012-2015 Daniel Nouri"""
        
        collected = {}
        prefix = '{}_'.format(name)

        params = vars(self)
        more_params = self.more_params

        for key, value in itertools.chain(params.items(), more_params.items()):
            if key.startswith(prefix):
                collected[key[len(prefix):]] = value

        return collected

    def _layer_name(self, layer_class, index):
        """This method has been adapted from the NeuralFit class in nolearn.
        https://github.com/dnouri/nolearn/blob/master/nolearn/lasagne/base.py
        Copyright (c) 2012-2015 Daniel Nouri"""
        return "{}{}".format(
            layer_class.__name__.lower().replace("layer", ""), index)
            
    def _initialize_function(self, fun_name, layers, input_layer_tuple, 
                             output_shape_marker, output_shape):
        """Function to build phi, psi and beta automatically from a 
        nolearn style network-as-list description.
        
        This method has been adapted from the NeuralFit class in nolearn.
        https://github.com/dnouri/nolearn/blob/master/nolearn/lasagne/base.py
        Copyright (c) 2012-2015 Daniel Nouri"""
            
        class Layers(OrderedDict):
            def __getitem__(self, key):
                if isinstance(key, int):
                    return list(self.values()).__getitem__(key)
                elif isinstance(key, slice):
                    items = list(self.items()).__getitem__(key)
                    return Layers(items)
                else:
                    return super(Layers, self).__getitem__(key)
        
            def keys(self):
                return list(super(Layers, self).keys())
        
            def values(self):
                return list(super(Layers, self).values())
            
        #self.__dict__[fun_name] = Layers()
        #fun_ = self.__dict__[fun_name]
        fun_ = Layers()
        
        # check if layers contains input layer; if not, create one
        user_input_layer = None
        for i, layer_def in enumerate(layers):
            if isinstance(layer_def[0], basestring):
                # The legacy format: ('name', Layer)
                layer_name, layer_factory = layer_def
                layer_kw = {'name': layer_name}
            else:
                # New format: (Layer, {'layer': 'kwargs'})
                layer_factory, layer_kw = layer_def
                layer_kw = layer_kw.copy()

            if issubclass(layer_factory, InputLayer):
                user_input_layer  = layer_factory
                break
        
        if isinstance(input_layer_tuple, list) or isinstance(input_layer_tuple, tuple):
            input_layer, input_layer_params = input_layer_tuple
        else:
            input_layer, input_layer_params = input_layer_tuple, None

        if (inspect.isclass(input_layer) and issubclass(input_layer, InputLayer))\
            or isinstance(input_layer, InputLayer):
            if user_input_layer is not None:
                # TODO check that the user provided input layer is compatible
                # with the one that the pattern expects
                # ok - we stick to the users input layer
                pass
            else:
                # push the input layer into the dictionary
                layers.insert(0, (input_layer, input_layer_params))
        else: # input_layer is output of another function 
            if user_input_layer is not None:
                # the user has provided an input layer. ignore it because
                # we use the functional input layer from the patern
                raise Exception("You have provided an input layer for %s," 
                    " but the pattern requires the input %s" % (fun_name, str(input_layer)))
            else:
                # push the input layer into the dictionary
                layers.insert(0, (input_layer, input_layer_params))
            
            
        # iterate through layers
        if isinstance(layers[0], Layer):
            # 'layers[0]' is already the output layer with type
            # 'lasagne.layers.Layer', so we only have to fill
            # 'fun_' and we're done:
            for i, layer in enumerate(get_all_layers(layers[0])):
                name = layer.name or self._layer_name(layer.__class__, i)
                fun_[name] = layer
                if self._get_params_for(name) != {}:
                    raise ValueError(
                        "You can't use keyword params when passing a Lasagne "
                        "instance object as the 'layers' parameter of "
                        "'Pattern'."
                        )
            return layers[0]

        # 'layers' are a list of '(Layer class, kwargs)', so
        # we'll have to actually instantiate the layers given the
        # arguments:
        layer = None
        for i, layer_def in enumerate(layers):
            
            if isinstance(layer_def[0], basestring):
                # The legacy format: ('name', Layer)
                layer_name, layer_factory = layer_def
                layer_kw = {'name': layer_name}
            else:
                # New format: (Layer, {'layer': 'kwargs'})
                layer_factory, layer_kw = layer_def
                if layer_kw is not None:
                    layer_kw = layer_kw.copy()

            layer_is_instance = False
            if layer_kw is None:
                # the passed object is a an expression or an object instance.
                # hence we don't have to build it later
                layer_is_instance = True
                layer_kw = {'name': layer_factory.name}

            if 'name' not in layer_kw:
                layer_kw['name'] = fun_name + "_" + self._layer_name(layer_factory, i)

            #more_params = self._get_params_for(layer_kw['name'])
            #layer_kw.update(more_params)

            if layer_kw['name'] in fun_:
                raise ValueError(
                    "Two layers with name {}.".format(layer_kw['name']))

            # Any layers that aren't subclasses of InputLayer are
            # assumed to require an 'incoming' paramter.  By default,
            # we'll use the previous layer as input:
            if not layer_is_instance and not issubclass(layer_factory, InputLayer):
                if 'incoming' in layer_kw:
                    layer_kw['incoming'] = fun_[
                        layer_kw['incoming']]
                elif 'incomings' in layer_kw:
                    layer_kw['incomings'] = [
                        fun_[nm] for nm in layer_kw['incomings']]
                else:
                    layer_kw['incoming'] = layer

            for attr in ('W', 'b'):
                if isinstance(layer_kw.get(attr), str):
                    name = layer_kw[attr]
                    layer_kw[attr] = getattr(fun_[name], attr, None)

            for k,v in layer_kw.items():
                if v == output_shape_marker:
                    #print ("%s triggered -> %s" % (output_shape_marker, str(output_shape)))
                    if output_shape is None:
                        raise ValueError("Cannot automatically set output shape (is None)"
                        " for %s - did you set all required shape variables" 
                        " in the constructor of the pattern?"
                        " (marker was: %s)" % (fun_name, output_shape_marker))
                    layer_kw[k] = output_shape

            if layer_is_instance:
                layer = layer_factory
                layer_wrapper = None
            else:
                try:
                    layer_wrapper = layer_kw.pop('layer_wrapper', None)
                    layer = layer_factory(**layer_kw)
                except TypeError as e:
                    msg = ("Failed to instantiate {} with args {}.\n"
                           "Maybe parameter names have changed?".format(
                               layer_factory, layer_kw))
                    raise Exception(TypeError(msg), e)
                    
            fun_[layer_kw['name']] = layer
            
            if layer_wrapper is not None:
                layer = layer_wrapper(layer)
                fun_["LW_%s" % layer_kw['name']] = layer
                
            layer._pattern_function = fun_name

        # we return the last layer as the representative of the function
        # as it's common in lasagne
        return layer

  
    def _create_target_objective(self, output=None, target=None):
        """
        Helper function to build the member variable target_loss.
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

    def _create_side_objective(self):
        """
        Helper function to build the member variable side_loss.
        """
        if self.side_loss is None:
            assert (self.input_var is not None)
            assert (self.side_var is not None)
            
            self.side_loss = self.get_side_objective(self.input_var, self.side_var)

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
        params = lasagne.layers.get_all_params(self.psi, **tags)
        if self.beta is not None:
            params += lasagne.layers.get_all_params(self.beta, **tags)
        params += lasagne.layers.get_all_params(self.phi, **tags) 
        return params
        
    def get_all_params(self, **tags):
        """Alias for get_params"""
        return self.get_params(**tags)
        
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
        return lasagne.layers.get_output(self.psi, inputs=input, **kwargs)

    def get_beta_output_for(self, input=None, **kwargs):
        if input is None:
            input = self.input_var
        return lasagne.layers.get_output(self.beta, inputs=input, **kwargs)

    def get_phi_output_for(self, input=None, **kwargs):
        if input is None:
            input = self.input_var
        return lasagne.layers.get_output(self.phi, inputs=input, **kwargs)

    def get_output_for_function(self, fun_or_fun_name, input, **kwargs):
        """
         Get the output for a pattern subfunction (i.e. phi, psi, beta) by
         setting the input to that subfunction manually.

         The problem is that if you apply lasagne.layers.get_output to, e.g., 
         psi, which gets phi as input, lasagne.layers.get_output will except
         the 'input' to be the input of phi, not the input of psi.
         
         Sometimes, it is desirable to set the input of psi manually.
         
         Note, that this differs from the behavior of get_phi_output_for, 
         get_psi_output_for and get_beta_output_for, which expect the initial
         input to the network (depending on the pattern, often phi's input).

        """
        
        if type(fun_or_fun_name) == str:
            fun = self.__dict__[fun_or_fun_name]
        else:
            fun = fun_or_fun_name
        
        last_input = input
        for l in self._get_all_function_layers(fun):
            last_input = l.get_output_for(last_input, **kwargs)
        return last_input    

    def training_loss(self, target_weight=0.5, side_weight=0.5, all_losses=False):
        """
        Compute the sum of the target and side info loss. Returns a theano expression.

        If all_losses is true, additionally to the summed loss the individual (weighted)
        losses are returned, too.

        Parameters
        ----------
        target_weight : float
            target weight
        side_weight : float
            side weight
        target_weight : float
            Default false, returns (loss) only.
            If true, returns tuple (loss, target_loss, side_loss)
        """

        # we need to gate because if we set one weight to 0., we might
        # also want to omit the involved theano variables; w/o the if-else
        # we get an "unconnected inputs" error in theano
        loss = 0.        
        tls, sls = np.nan, np.nan
        
        if target_weight > 0.:
            tls = target_weight * self.target_loss
            loss += tls
        if side_weight > 0.:
            sls = side_weight * self.side_loss
            loss += sls
            
        if all_losses:
            return loss, tls, sls        
            
        return loss

    def save(self, fn):
        """
        Save your pattern's weights in a tar file containing npz files.
        You can then use `load` to recreate the pattern from this file.

        Parameters
        ----------
        fn : str
            file name
        """
        # tar to one file
        tmp_files = []
        with tarfile.open(fn + ".tar", mode='w') as out:
            # use lasagne style parameter storage to avoid CUDA vs. non-CUDA
            # theano issue 
            phi_pval = lasagne.layers.get_all_param_values(self.phi)
            phi_fn = fn+"_phi.npz"
            np.savez(phi_fn, *phi_pval)
            out.add (phi_fn)
            tmp_files.append(phi_fn)
        
            psi_pval = lasagne.layers.get_all_param_values(self.psi)
            psi_fn = fn+"_psi.npz"
            np.savez(psi_fn, *psi_pval)
            out.add (psi_fn)
            tmp_files.append(psi_fn)

            if self.beta is not None:
                beta_pval = lasagne.layers.get_all_param_values(self.beta)
                beta_fn = fn+"_beta.npz"
                np.savez(beta_fn, *beta_pval)
                out.add (beta_fn)
                tmp_files.append(beta_fn)
    
        for d in tmp_files:
            try:
                os.unlink(d)
            except:
                pass
                
    def load(self, fn):
        """
        Assuming you have initialized the pattern exactly as it
        was pickled in the file 'fn', you can restore all function
        parameters using this function.

        Parameters
        ----------
        fn : str
            file name
        """
        
        fn_split = os.path.splitext(fn)
        
        assert (fn_split[-1] == ".tar")
        fun_names = ['phi', 'psi', 'beta']
        npz_files_loaded = {}
        
        with tarfile.open(fn, mode='r') as t:
            for m in t.getmembers():
                for fun_name in fun_names:
                    if fun_name in m.name:
                        #print ("extracting" + str(m))
                        t.extract(m)
                        npz_files_loaded[fun_name] = m.name

        for fun_name, fun_npz in npz_files_loaded.items():
            with np.load(fun_npz) as f:
                param_values = [f['arr_%d' % i] for i in range(len(f.files))]
                lasagne.layers.set_all_param_values(self.__dict__[fun_name], param_values)

        for _,d in npz_files_loaded.items():
            try:
                os.unlink(d)
            except:
                pass
