#!/usr/bin/env python
'''
Main file for running a mbs simulation.
'''

from floripy import flowfield
from floripy import hydrodynamics
from floripy.rbd import mbs
from floripy.rbd import timestepper
from floripy.file_formats import yamlio


def run(fn_options):
    '''
    options : dict

    '''
    options = yamlio.read(fn_options)
    model = mbs.Mbs(**options['model'])
    ffield = flowfield.create(**options['flowfield'])
    hydrodyn = hydrodynamics.create(model, ffield,
                    options['hydrodynamics'])
    ts = timestepper.Timestepper(model, hydrodyn, ffield,
            **options['timestepper'])
    ts.run()
