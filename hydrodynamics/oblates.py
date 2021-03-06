#!/usr/bin/env python

import math
import numpy as np
from floripy.mathutils.linalg import perm_tensor
from .hydrodynamicsbase import HydrodynamicsBase


class Oblates_hydrodynamics(HydrodynamicsBase):
    def __init__(self, model, flowfield, kwargs):
        self._model = model
        self._num_bodies = self._model.num_bodies
        self._all_oblate_a = model.get_all_oblate_a()
        self._all_oblate_c = model.get_all_oblate_c()
        self._calc_rf()
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


    def _calc_rf(self):
        self._rf = {'XA':np.zeros((self._num_bodies,)),
                    'YA':np.zeros((self._num_bodies,)),
                    'XC':np.zeros((self._num_bodies,)),
                    'YC':np.zeros((self._num_bodies,)),
                    'YH':np.zeros((self._num_bodies,))}

        for i in range(self._num_bodies):
            a = self._all_oblate_a[i]
            c = self._all_oblate_c[i]
            if c == 'zero':
                self._rf['XA'][i] = 8.0/(3*math.pi)
                self._rf['YA'][i] = 16.0/(9*math.pi)
                self._rf['XC'][i] = 4.0/(3*math.pi)
                self._rf['YC'][i] = 4.0/(3*math.pi)
                self._rf['YH'][i] = -4.0/(3*math.pi)
            else:
                e = math.sqrt((a+c)*(a-c))/a
                e2 = e*e
                e3 = e2*e
                e5 = e2*e3
                one_m_e2 = 1.0 - e2
                two_m_e2 = 2.0 - e2
                sqrt_one_m_e2 = math.sqrt(one_m_e2)

                C = math.atan(e/sqrt_one_m_e2)
                self._rf['XA'][i] = (4.0/3)*e3/((2*e2-1)*C + e*sqrt_one_m_e2)
                self._rf['YA'][i] = (8.0/3)*e3/((2*e2+1)*C - e*sqrt_one_m_e2)
                self._rf['XC'][i] = (2.0/3)*e3/(C - e*sqrt_one_m_e2)
                self._rf['YC'][i] = (2.0/3)*e3*two_m_e2/(e*sqrt_one_m_e2 - (1-2*e2)*C)
                self._rf['YH'][i] = (-2.0/3)*e5/(e*sqrt_one_m_e2 - (1-2*e2)*C)


    def _calc_grm(self):
        I = np.identity(3)
        epsilon = perm_tensor()
        rm = np.zeros((6,15))
        self._all_director = self._model.get_all_director()
        for i in range(self._num_bodies):
            a = self._all_oblate_a[i]
            a3 = a**3
            XA = self._rf['XA'][i]
            YA = self._rf['YA'][i]
            XC = self._rf['XC'][i]
            YC = self._rf['YC'][i]
            YH = self._rf['YH'][i]
            d = self._all_director[i]
            dd = np.outer(d,d)

            #Submatrix C in the resistance matrix
            rm[0:3,0:3] = 8*math.pi*a3*self._viscosity*(XC*dd + YC*(I-dd))

            #Matrix representaion of Htilde in the resistance matrix
            rm[0:3,6:15] = (-8*math.pi*a3*self._viscosity*YH
                                *np.einsum('ijl,l,k',
                                epsilon, d, d)).reshape((3,9))

            #Submatrix A in the resistance matrix
            rm[3:6,3:6] = 6*math.pi*a*self._viscosity*(XA*dd + YA*(I-dd))

            self._grm[6*i:6*i+6,15*i:15*i+15] = rm


    def _calc_gmm(self):
        '''
        Returns the mobility matrix for a set of point particles.
        '''
        raise NotImplementedError


    def update(self, time):
        if hasattr(self, '_grm'):
            self._calc_grm()
        if hasattr(self, '_gmm'):
            self._calc_gmm()
