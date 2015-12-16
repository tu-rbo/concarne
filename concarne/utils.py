# -*- coding: utf-8 -*-

__all__ = ["isfunction"]

def isfunction(f):
    return hasattr(f, '__call__')