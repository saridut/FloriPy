#!/usr/bin/env python

from floripy.file_formats import yamlio

class SimOptions(object):
    def __init__(self):
        #Set reasonable defaults
        self._options = {'model': None, 'timestepper': None,
                'hydrodynamics': None, 'flowfield': None}

        self._options['model'] = {
                'fn_model': 'model.yaml',
                'fn_out': 'history.h5',
                'draw_graph': False,
                'rooted': False,
                'pullback': True,
                'pullback_radius': 10.0,
                'restart': False,
                'time_start': 0.0   #Redundant??
                }

        self._options['timestepper'] = {
                'time_start': 0.0,
                'time_end': 1.0,
                'time_stepsize': 0.001,
                'time_write_interval': 0.25,
                'time_pullback_interval': 20,
                'time_print_interval': 100,
                'time_flow_reversal': 5.0,
                'odeopts': None
                }

        self._options['timestepper']['odeopts'] = {'name': 'dop853',
                'nsteps': 5000, 'atol': 1.E-7, 'rtol': 1.E-7}

        self._options['hydrodynamics'] = {
                'viscosity': 1.0,
                'form': 'resistance',
                'hydrodynamics_type': 'ellipsoids'
                }


    @classmethod
    def from_file(cls, fn):
        '''
        Read options from a file.

        '''
        raise NotImplementedError
        options = yamlio.read(fn)
        sop = cls()
        sop.options.update(options)
        return sop


    def to_file(self, fn):
        if self._options['flowfield'] is None:
            raise ValueError('flowfield must be specified!')
        yamlio.write(self._options, fn, default_flow_style=False, indent=4)


    def set_fn_model(self, fn_model):
        self._options['model']['fn_model'] = fn_model


    def set_restart(self, restart):
        self._options['model']['restart'] = restart


    def set_fn_out(self, fn_out):
        self._options['model']['fn_out'] = fn_out


    def set_timestepper_options(self, **kwargs):
        self._options['timestepper'].update(kwargs)
        if 'time_start' in kwargs:
            self._options['model']['time_start'] = kwargs['time_start']


    def set_shear_flow(self, flow_dir='x', gradient_dir='y'):
        self._options['flowfield'] = {'flow_type': 'shear',
                'flow_dir': flow_dir, 'gradient_dir': gradient_dir,
                'shear_rate': 1.0}


    def set_planar_extensional_flow(self, ext_dir='x', comp_dir='y'):
        self._options['flowfield'] = {'flow_type': 'planar_extension',
                'plane': ext_dir + comp_dir, 'strain_rate': 1.0}


    def set_uniaxial_extensional_flow(self, ext_dir='x'):
        self._options['flowfield'] = {'flow_type': 'uniaxial_extension',
                'axis': ext_dir, 'strain_rate': 1.0}


    def set_biaxial_extensional_flow(self, comp_plane='yz'):
        self._options['flowfield'] = {'flow_type': 'biaxial_extension',
                'plane': comp_plane, 'strain_rate': 1.0}


    def set_linear_flow(self, U, Omega, E):
        self._options['flowfield'] = {'flow_type': 'linear',
                'U': U.tolist(), 'Omega': Omega.tolist(), 'E': E.tolist()}


    def set_hydrodynamics_type(self, key):
        self._options['hydrodynamics']['hydrodynamics_type'] = key
