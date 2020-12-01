#!/usr/bin/env python

from .shear import Shear_flow
from .planar import Planar_extensional_flow
from .uniaxial import Uniaxial_extensional_flow
from .biaxial import Biaxial_extensional_flow
from .linear import Linear_flow


def create(**kwargs):
    '''Creates a flow field.

    Parameters
    ----------
    kwargs : dict
        Keyword arguments specifying the details of a particular kind of flow
        field.
        flow_type : {'shear', planar_extension', 'uniaxial_extension',
            'biaxial_extension', 'linear'}
            Specifies the kind of flow field.

        The keys and their corresponding values depend on `flow_type`.
        The allowed keys for each value of `flow_type` are as follows.

        If `flow_type` = 'shear':
            'shear_rate' : float
            'flow_dir' : {'x', 'y', 'z'}
            'gradient_dir' : {'x', 'y', 'z'}
        If `flow_type` = 'planar_extension':
            'strain_rate' : float
            'plane' : string
                Composed of two characters, the first specifying the expansion
                axis, and the second specifying the compression axis. For
                example, 'yz' specifies the expansion axis is 'y' and the
                compression axis is 'z'.
        If `flow_type` = 'uniaxial_extension':
            'strain_rate' : float
            'axis' :  {'x', 'y', 'z'}
                Specifies the expansion axis.
        If `flow_type` = 'biaxial_extension':
            'strain_rate' : float
            'plane' :  string
                Composed of two characters specifying the two axes defining
                the compression plane. The order of the characters does not
                matter.
        If `flow_type` = 'linear':
            'U' : list of floats
                Translational velocity
            'Omega' : list of floats
                Angular velocity
            'E' : list of lists of floats
                Rate of strain tensor

    Returns
    -------
    flowfield
        Instance of flowfield object

    Examples
    --------
    1. Create a shear flow field.
       >>> import flowfield
       >>> kwargs = {'flow_type': 'shear', 'shear_rate': 0.25, 'flow_dir': 'x', 'gradient_dir': 'y'}
       >>> sf = flowfield.create(**kwargs)

    2. Create a linear flow field.
       >>> import flowfield
       >>> kwargs = {'flow_type': 'linear', 'U': [0.25, 0.0, 0.0], 'Omega': [0.0, 0.0, 0.25],
       ... 'E': [[0.0, 0.0, 0.0],[0.0, 0.0, 0.0],[0.0, 0.0, 0.0]]}
       >>> lf = flowfield.create(**kwargs)

    '''
    classes = {'shear': Shear_flow,
                'planar_extension': Planar_extensional_flow,
                'uniaxial_extension': Uniaxial_extensional_flow,
                'biaxial_extension': Biaxial_extensional_flow,
                'linear': Linear_flow}
    flow_type = kwargs.pop('flow_type')
    out = classes[flow_type](**kwargs)
    return out
