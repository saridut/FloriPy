#!/usr/bin/env python

import math
import mpmath
import numpy as np

from floripy.mathutils.xform import shift_tensor2_dcm, shift_tensor3_dcm
from .hydrodynamicsbase import HydrodynamicsBase


class Ellipsoids_hydrodynamics(HydrodynamicsBase):
    def __init__(self, model, flowfield, kwargs):
        self._model = model
        self._num_bodies = self._model.num_bodies
        self._all_ellipsoid_a = model.get_all_ellipsoid_a()
        self._all_ellipsoid_b = model.get_all_ellipsoid_b()
        self._all_ellipsoid_c = model.get_all_ellipsoid_c()
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
        self._rf = np.zeros(self._num_bodies,
                        dtype=[('A','f8',(3,3)), ('C','f8',(3,3)),
                                ('H_tilde','f8',(3,3,3))])

        for i in range(self._num_bodies):
            a = self._all_ellipsoid_a[i]
            b = self._all_ellipsoid_b[i]
            c = self._all_ellipsoid_c[i]

            asq = a**2
            bsq = 0.0 if b == 'zero' else b**2
            csq = c**2

            #The integrals alpha_i
            asq_alpha_a = (2*asq/3.0)*mpmath.elliprj(asq, bsq, csq, asq)
            csq_alpha_c = (2*csq/3.0)*mpmath.elliprj(asq, bsq, csq, csq)
            if b == 'zero':
                bsq_alpha_b = 0.0
            else:
                bsq_alpha_b = (2*bsq/3.0)*mpmath.elliprj(asq, bsq, csq, bsq)

            #The integral chi 
            chi = 2*mpmath.elliprf(asq, bsq, csq)

            sxp = 16.0*math.pi
            sxp3 = sxp/3.0

            #Submatrix A in the resistance matrix
            self._rf[i]['A'][0,0] = sxp/(chi + asq_alpha_a)
            self._rf[i]['A'][1,1] = sxp/(chi + bsq_alpha_b)
            self._rf[i]['A'][2,2] = sxp/(chi + csq_alpha_c)

            #Submatrix C in the resistance matrix
            self._rf[i]['C'][0,0] = sxp3*(bsq+csq)/(bsq_alpha_b + csq_alpha_c)
            self._rf[i]['C'][1,1] = sxp3*(csq+asq)/(asq_alpha_a + csq_alpha_c)
            self._rf[i]['C'][2,2] = sxp3*(asq+bsq)/(asq_alpha_a + bsq_alpha_b)

            #Tensor (third order) H_tilde in the resistance matrix
            self._rf[i]['H_tilde'][0,1,2] =  sxp3*bsq/(bsq_alpha_b + csq_alpha_c)
            self._rf[i]['H_tilde'][0,2,1] = -sxp3*csq/(bsq_alpha_b + csq_alpha_c)
            self._rf[i]['H_tilde'][1,2,0] =  sxp3*csq/(asq_alpha_a + csq_alpha_c)
            self._rf[i]['H_tilde'][1,0,2] = -sxp3*asq/(asq_alpha_a + csq_alpha_c)
            self._rf[i]['H_tilde'][2,0,1] =  sxp3*asq/(asq_alpha_a + bsq_alpha_b)
            self._rf[i]['H_tilde'][2,1,0] = -sxp3*bsq/(asq_alpha_a + bsq_alpha_b)


    def _calc_grm(self):
        shifters = self._model.get_all_ellipsoid_shifters()
        for i in range(self._num_bodies):
            shifter = shifters[i]
            dcm = shifter.T
            rm = np.zeros((6,15))
            A = self._rf[i]['A']
            C = self._rf[i]['C']
            H_tilde = self._rf[i]['H_tilde']
            #Submatrix C in the resistance matrix
            rm[0:3,0:3] = shift_tensor2_dcm(C, dcm, forward=False)

            #Matrix representaion of Htilde in the resistance matrix
            H_tilde = shift_tensor3_dcm(H_tilde, dcm, forward=False)
            rm[0:3,6:15] = H_tilde.reshape((3,9)) 

            #Submatrix A in the resistance matrix
            rm[3:6,3:6] = shift_tensor2_dcm(A, dcm, forward=False)

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
