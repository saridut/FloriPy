#!/usr/bin/env python

import math
import numpy as np
from .flowfieldbase import FlowfieldBase

class Uniaxial_extensional_flow(FlowfieldBase):

    def __init__(self, **kwargs):
        self.set_property('strain_rate', kwargs['strain_rate'])
        self.set_property('axis', kwargs['axis'])


    def set_property(self, property_name, property_value):
        '''
        Set a single property.
        '''
        if property_name == 'strain_rate':
            self._strain_rate = property_value
        elif  property_name == 'axis':
            self._exp_axis = {'x':0, 'y':1, 'z':2}[property_value]
            self._comp_axis = [0, 1, 2]
            self._comp_axis.remove(self._exp_axis)
        else:
            raise ValueError('Unknown property', property_name)


    def get_property(self, property_name):
        '''
        Get a single property, the strain_rate. These are merely convenience
        functions, the axes cannot be changed. The flow is always axis-aligned,
        for more control, use the general linear flowfield.
        '''
        if property_name == 'strain_rate':
            return self._strain_rate
        else:
            raise ValueError('Unknown property', property_name)


    def get_velocity(self, points, out=None):
        '''
        Calculates the velocity at the points. If out is given, it will contain
        the calculated velocity field. Make sure <points> is a Nx3 array, and so
        is <out>.
        '''
        N = points.shape[0]
        if out is None:
            out = np.zeros((3*N,))
        out[self._exp_axis::3] = (self._strain_rate
                                    *np.ravel(points)[self._exp_axis::3])
        out[self._comp_axis[0]::3] = (-0.5*self._strain_rate
                                    *np.ravel(points)[self._comp_axis[0]::3])
        out[self._comp_axis[1]::3] = (-0.5*self._strain_rate
                                    *np.ravel(points)[self._comp_axis[1]::3])
        return out


    def get_velocity_meshgrid(self, x_mg, y_mg, z_mg, u_mg=None,
                    v_mg=None, w_mg=None):
        '''
        Returns the velocity on a meshgrid.
        '''
        if u_mg is None:
            u_mg = np.zeros_like(x_mg)

        if v_mg is None:
            v_mg = np.zeros_like(y_mg)

        if w_mg is None:
            w_mg = np.zeros_like(z_mg)

        u_mg = 0.0
        v_mg = 0.0
        w_mg = 0.0

        if self._exp_axis == 0:
            u_mg[:,:,:] = self._strain_rate*x_mg
            v_mg[:,:,:] = -0.5*self._strain_rate*y_mg
            w_mg[:,:,:] = -0.5*self._strain_rate*z_mg
        elif self._exp_axis == 1:
            u_mg[:,:,:] = -0.5*self._strain_rate*x_mg
            v_mg[:,:,:] = self._strain_rate*y_mg
            w_mg[:,:,:] = -0.5*self._strain_rate*z_mg
        elif self._exp_axis == 2:
            u_mg[:,:,:] = -0.5*self._strain_rate*x_mg
            v_mg[:,:,:] = -0.5*self._strain_rate*y_mg
            w_mg[:,:,:] = self._strain_rate*z_mg

        return u_mg, v_mg, w_mg


    def get_streaming_velocity(self, points=None, out=None):
        streaming_velocity = np.zeros((3,))
        if points is None:
            return streaming_velocity
        else:
            N = points.shape[0]
            if out is None:
                out = np.zeros((3*N,))
            out[:] = np.tile(streaming_velocity,N)
        return out


    def get_angular_velocity(self, points=None, out=None):
        angular_velocity = np.zeros((3,))
        if points is None:
            return angular_velocity
        else:
            N = points.shape[0]
            if out is None:
                out = np.zeros((3*N,))
            out = 0.0
        return out


    def get_strainrate_tensor(self, points=None, out=None):
        strainrate_tensor = np.zeros((3,3))
        strainrate_tensor[self._exp_axis,
                            self._exp_axis] = self._strain_rate
        strainrate_tensor[self._comp_axis[0],
                            self._comp_axis[0]] = -0.5*self._strain_rate
        strainrate_tensor[self._comp_axis[1],
                            self._comp_axis[1]] = -0.5*self._strain_rate
        if points is None:
            return strainrate_tensor
        else:
            N = points.shape[0]
            if out is None:
                out = np.zeros((9*N,))
            out[:] = np.tile(strainrate_tensor.ravel(),N)
        return out
