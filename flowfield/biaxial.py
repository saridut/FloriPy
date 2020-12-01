#!/usr/bin/env python

import math
import numpy as np
from .uniaxial import Uniaxial_extensional_flow

class Biaxial_extensional_flow(Uniaxial_extensional_flow):
    def __init__(self, **kwargs):
        if 'strain_rate' in kwargs:
            kwargs['strain_rate'] = -kwargs['strain_rate']
        if 'plane' in kwargs:
            plane = kwargs.pop('plane')
            kwargs['axis'] = 'xyz'.strip(plane)
        super(Biaxial_extensional_flow, self).__init__(**kwargs)

