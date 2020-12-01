#!/usr/bin/env python

import math
import numpy as np
from .flowfieldbase import FlowfieldBase


class Shear_flow(FlowfieldBase):

    def __init__(self, **kwargs):
        self.set_property('shear_rate', kwargs['shear_rate'])
        self.set_property('flow_dir', kwargs['flow_dir'])
        self.set_property('gradient_dir', kwargs['gradient_dir'])


    def set_property(self, property_name, property_value):
        '''
        Set a single property.
        '''
        if property_name == 'shear_rate':
            self._shear_rate = property_value
        elif  property_name == 'flow_dir':
            self._flow_dir = {'x':0, 'y':1, 'z':2}[property_value]
        elif  property_name == 'gradient_dir':
            self._gradient_dir = {'x':0, 'y':1, 'z':2}[property_value]
        else:
            raise ValueError('Unknown property', property_name)
        if hasattr(self,'_flow_dir') and hasattr(self, '_gradient_dir'):
            assert self._flow_dir != self._gradient_dir


    def get_velocity(self, points, out=None):
        '''
        Calculates the velocity at the points. If out is given, it will contain
        the calculated velocity field. 
        <points> is a Nx3 array.
        <out> is a numpy vector of size 3*N.
        If <out> is not given, returns numpy vector of size 3*N.
        '''
        N = points.shape[0]
        if out is None:
            out = np.zeros((3*N,))
        out[self._flow_dir::3] = (self._shear_rate
                                    *np.ravel(points)[self._gradient_dir::3])
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

        if self._flow_dir == 0:
            if self._gradient_dir == 1:
                u_mg[:,:,:] = self._shear_rate*y_mg
            if self._gradient_dir == 2:
                u_mg[:,:,:] = self._shear_rate*z_mg
        elif self._flow_dir == 1:
            if self._gradient_dir == 0:
                v_mg[:,:,:] = self._shear_rate*x_mg
            if self._gradient_dir == 2:
                v_mg[:,:,:] = self._shear_rate*z_mg
        elif self._flow_dir == 2:
            if self._gradient_dir == 0:
                w_mg[:,:,:] = self._shear_rate*x_mg
            if self._gradient_dir == 1:
                w_mg[:,:,:] = self._shear_rate*y_mg

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
        #Angular velocity is half of the vorticity vector.
        #The vorticity vector is the curl of the velocity field.
        axes = [0, 1, 2]
        axes.remove(self._flow_dir)
        axes.remove(self._gradient_dir)
        vorticity_dir = axes[0]
        angular_velocity = np.zeros((3,))
        angular_velocity[vorticity_dir] = -0.5*self._shear_rate
        if points is None:
            return angular_velocity
        else:
            N = points.shape[0]
            if out is None:
                out = np.zeros((3*N,))
            out[:] = np.tile(angular_velocity,N)
        return out


    def get_strainrate_tensor(self, points=None, out=None):
        strainrate_tensor = np.zeros((3,3))
        strainrate_tensor[0,1] = 0.5*self._shear_rate
        strainrate_tensor[1,0] = 0.5*self._shear_rate
        if points is None:
            return strainrate_tensor
        else:
            N = points.shape[0]
            if out is None:
                out = np.zeros((9*N,))
            out[:] = np.tile(strainrate_tensor.ravel(),N)
        return out

