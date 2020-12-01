#!/usr/bin/env python

import math
import numpy as np
from abc import ABCMeta, abstractmethod


class FlowfieldBase:
    __metaclass__ = ABCMeta

    @abstractmethod
    def set_property(self, property_name, property_value):
        pass

#   @abstractmethod
#   def get_property(self, property_name):
#       pass

    @abstractmethod
    def get_velocity(self, points, out):
        pass

    @abstractmethod
    def get_velocity_meshgrid(self, x_mg, y_mg, z_mg, u_mg, v_mg, w_mg):
        pass

    @abstractmethod
    def get_streaming_velocity(self, points, out):
        pass

    @abstractmethod
    def get_angular_velocity(self, points, out):
        pass

    @abstractmethod
    def get_strainrate_tensor(self, points, out):
        pass

    def get_uoe(self, points, arrangement='interlaced', out=None):
        '''
        Returns a vector containing streaming velocity, angular velocity, and strain
        tensor. points is an Nx3 numpy array. out is a vector.
        '''
        N = points.shape[0]
        if out is None:
            out = np.zeros((15*N,))
        if arrangement == 'interlaced':
            angular_velocity = self.get_angular_velocity()
            strainrate_tensor = self.get_strainrate_tensor().ravel()
            velocity = self.get_velocity(points=points)
            for i in range(N):
                out[15*i:15*i+3] = velocity[3*i:3*i+3]
                out[15*i+3:15*i+6] = angular_velocity
                out[15*i+6:15*i+15] = strainrate_tensor
        elif arrangement == 'stacked':
            out[0:3*N] = self.get_velocity(
                                points=points, out=out[0:3*N])
            out[3*N:6*N] = self.get_angular_velocity(
                                points=points, out=out[3*N:6*N])
            out[6*N:15*N] = self.get_strainrate_tensor(
                                points=points, out=out[6*N:15*N])
        return out


    def get_oue(self, points, arrangement='interlaced', out=None):
        '''
        Returns a vector containing the angular velocity, streaming velocity, and strain
        tensor. points is an Nx3 numpy array. out is a vector.
        '''
        N = points.shape[0]
        if out is None:
            out = np.zeros((15*N,))
        if arrangement == 'interlaced':
            angular_velocity = self.get_angular_velocity()
            strainrate_tensor = self.get_strainrate_tensor().ravel()
            velocity = self.get_velocity(points=points)
            for i in range(N):
                out[15*i:15*i+3] = angular_velocity
                out[15*i+3:15*i+6] = velocity[3*i:3*i+3]
                out[15*i+6:15*i+15] = strainrate_tensor
        elif arrangement == 'stacked':
            out[0:3*N] = self.get_angular_velocity(
                            points=points, out=out[0:3*N])
            out[3*N:6*N] = self.get_velocity(
                                points=points, out=out[3*N:6*N])
            out[6*N:15*N] = self.get_strainrate_tensor(
                                points=points, out=out[6*N:15*N])
        return out

