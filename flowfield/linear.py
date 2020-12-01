#!/usr/bin/env python

import math
import numpy as np
from .flowfieldbase import FlowfieldBase

class Linear_flow(FlowfieldBase):

    def __init__(self, **kwargs):
        self.set_property('U', kwargs['U'])
        self.set_property('Omega', kwargs['Omega'])
        self.set_property('E', kwargs['E'])


    def set_property(self, property_name, property_value):
        '''
        Set a single property.
        '''
        if property_name == 'U':
            self._U = np.asarray(property_value)
        elif  property_name == 'Omega':
            self._Omega = np.asarray(property_value)
        elif  property_name == 'E':
            self._E = np.asarray(property_value)
        else:
            raise ValueError('Unknown property', property_name)


    def reverse_flow(self):
        #TODO: Fix handling of reversal in cases when shear rate is updated,
        #etc.
        raise NotImplementedError
        self._U = -self._U
        self._Omega = -self._Omega
        self._E = -self._E


    def get_velocity(self, points, out=None):
        '''
        Calculates the velocity at the points. If out is given, it will contain
        the calculated velocity field. Make sure <points> is a Nx3 array, and so
        is <out>.
        '''
        N = points.shape[0]
        if out is None:
            out = np.zeros((3*N,))
        for i in range(N):
            out[3*i:3*i+3] = (self._U
                            + np.cross(self._Omega, points[i,:])
                            + np.dot(self._E, points[i,:]))
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

        u_mg[:,:,:] = (self._U[0]
                        + self._Omega[1]*z_mg - self._Omega[2]*y_mg
                        + self._E[0,0]*x_mg + self._E[0,1]*y_mg
                        + self._E[0,2]*z_mg)

        v_mg[:,:,:] = (self._U[1] 
                        - self._Omega[0]*z_mg + self._Omega[2]*x_mg
                        + self._E[1,0]*x_mg + self._E[1,1]*y_mg
                        + self._E[1,2]*z_mg)

        w_mg[:,:,:] = (self._U[2] + 
                        + self._Omega[0]*y_mg - self._Omega[1]*x_mg
                        + self._E[2,0]*x_mg + self._E[2,1]*y_mg
                        + self._E[2,2]*z_mg)

        return u_mg, v_mg, w_mg


    def get_streaming_velocity(self, points=None, out=None):
        streaming_velocity = self._U
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
        angular_velocity = self._Omega
        if points is None:
            return angular_velocity
        else:
            N = points.shape[0]
            if out is None:
                out = np.zeros((3*N,))
            out[:] = np.tile(angular_velocity,N)
        return out


    def get_strainrate_tensor(self, points=None, out=None):
        strainrate_tensor = self._E
        if points is None:
            return strainrate_tensor
        else:
            N = points.shape[0]
            if out is None:
                out = np.zeros((9*N,))
            out[:] = np.tile(strainrate_tensor.ravel(),N)
        return out


