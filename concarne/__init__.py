try:
    install_instr = """
Please make sure you install a recent enough version of Theano and lasagne. 
http://lasagne.readthedocs.org/en/latest/user/installation.html"""

    import lasagne
except ImportError:  # pragma: no cover
    raise ImportError("Could not import lasagne." + install_instr)
else:
    del install_instr
    del lasagne


#from . import nonlinearities


__version__ = "0.1.dev1"