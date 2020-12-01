#!/usr/bin/env python

import math
import numpy as np
from .hydrodynamicsbase import HydrodynamicsBase


class Spheres_hydrodynamics(HydrodynamicsBase):
    def __init__(self, model, flowfield, kwargs):
        self._model = model
        self._num_bodies = self._model.num_bodies
        self._all_radius = model.get_all_radius()
        self._flowfield = flowfield
        self._viscosity = kwargs['viscosity']
        form = kwargs['form']
        if form == 'resistance':
            self._grm = np.zeros((6*self._num_bodies,15*self._num_bodies))
            self._calc_grm()
        elif form == 'mobility':
            self._gmm = np.zeros((6*self._num_bodies,15*self._num_bodies))
            self._calc_gmm()
        else:
            raise ValueError('Unkown form ', form)


    def _calc_grm(self):
        I = np.identity(3)
        for i in range(self._num_bodies):
            a = self._all_radius[i]
            a3 = a**3
            #Note: Htilde is non-zero, but E_inf is symmetric and hence
            #Htilde E_inf must be zero. So setting Htilde to zero here.
            self._grm[6*i:6*i+3,15*i:15*i+3] = 8*math.pi*a3*self._viscosity*I
            self._grm[6*i+3:6*i+6,15*i+3:15*i+6] = 6*math.pi*a*self._viscosity*I


    def _calc_gmm(self):
        '''
        Returns the grand mobility matrix for a set of spheres.
        '''
        raise NotImplementedError


    def update(self, time):
        '''
        Update the flowfield if required.
        '''
        pass
