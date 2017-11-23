# -*- coding: utf-8 -*-
"""
Created on Thu Nov 23 16:11:55 2017

@author: Jesus
"""

import numpy as np

from contextlib import contextmanager
@contextmanager
def print_context(*args, **kwargs):
    original = np.get_printoptions()
    np.set_printoptions(*args, **kwargs)
    try:
        yield
    finally:
        np.set_printoptions(**original)