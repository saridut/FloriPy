#!/usr/bin/env python

#from . import transform
from .spheres import Spheres_hydrodynamics
from .prolates import Prolates_hydrodynamics
from .oblates import Oblates_hydrodynamics
from .ellipsoids import Ellipsoids_hydrodynamics


def create(model, flowfield, kwargs):
    hydrodynamics_type = kwargs.pop('hydrodynamics_type')
    classes = {'spheres': Spheres_hydrodynamics,
                'prolates': Prolates_hydrodynamics,
                'oblates': Oblates_hydrodynamics,
                'ellipsoids': Ellipsoids_hydrodynamics}
    hydrodynamics = classes[hydrodynamics_type](model, flowfield, kwargs)
    return hydrodynamics
