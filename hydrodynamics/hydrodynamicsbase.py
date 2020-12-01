#!/usr/bin/env python

from abc import ABCMeta, abstractmethod
import numpy as np
from scipy import linalg as sla


class HydrodynamicsBase:
    __metaclass__ = ABCMeta

    @abstractmethod
    def _calc_grm(self):
        pass

    @abstractmethod
    def _calc_gmm(self):
        pass

    @abstractmethod
    def update(self, time):
        pass

    def grmvec(self, vec, out=None):
        '''
        Returns the product of the grand resistance matrix with another numpy
        vector.
        '''
        if out is None:
            out = np.zeros((self._grm.shape[0],))
        out = np.dot(self._grm, vec)
        return out


    def gmmvec(self, vec, out=None):
        '''
        Returns the product of the grand mobility matrix with another numpy
        vector.
        '''
        if out is None:
            out = np.zeros((self._gmm.shape[0],))
        out = np.dot(self._gmm, vec, out=out)
        return out


    def grmvel(self, out=None):
        '''
        Returns the product of the grand resistance matrix with the fluid
        velocity at the given points.
        '''
        points = np.array(self._model.get_coms())
        vec = self._flowfield.get_oue(points, arrangement='interlaced', out=None)
        return self.grmvec(vec, out=out)


    def gmmvel(self, out=None):
        '''
        Returns the product of the grand mobility matrix with the fluid
        velocity at the given points.
        '''
        points = np.array(self._model.get_coms())
        vec = self._flowfield.get_oue(points, arrangement='interlaced', out=None)
        return self.gmmvec(vec, out=out)


    def getrans_grm_ge(self, ge, out=None):
        '''
        Returns the product A'RA, where A is a matrix, R is the resistance
        matrix for force and torque, and A' is the transpose of A.
        '''
        if out is None:
            n = ge.shape[1]
            out = np.zeros((n,n))
        grm = np.zeros((6*self._num_bodies,6*self._num_bodies))
        for i in range(self._num_bodies):
            grm[6*i:6*i+6, 6*i:6*i+6] = self._grm[6*i:6*i+6, 15*i:15*i+6]
        out = np.dot(ge.T, np.dot(grm, ge), out=out)
        return out


    def getrans_gmm_ge(self, ge, out=None):
        '''
        Returns the product A'MA, where A is a matrix, M is the mobility
        matrix for force and torque, and A' is the transpose of A.
        '''
        if out is None:
            n = ge.shape[1]
            out = np.zeros((n,n))
        gmm = np.zeros((6*self._num_bodies,6*self._num_bodies))
        for i in range(self._num_bodies):
            gmm[6*i:6*i+6, 6*i:6*i+6] = self._gmm[6*i:6*i+6, 15*i:15*i+6]
        out = np.dot(ge.T, np.dot(gmm, ge), out=out)
        return out


    def grm_cholesky_decomp(self):
        '''
        Returns the Cholesky decomposition of the grand resistance matrix.
        '''
        grm = np.zeros((6*self._num_bodies,6*self._num_bodies))
        for i in range(self._num_bodies):
            grm[6*i:6*i+6, 6*i:6*i+6] = self._grm[6*i:6*i+6, 15*i:15*i+6]

        return sla.cholesky(grm, lower=True)


    def gmm_cholesky_decomp(self):
        '''
        Returns the Cholesky decomposition of the grand mobility matrix.
        '''
        gmm = np.zeros((6*self._num_bodies,6*self._num_bodies))
        for i in range(self._num_bodies):
            gmm[6*i:6*i+6, 6*i:6*i+6] = self._gmm[6*i:6*i+6, 15*i:15*i+6]

        return sla.cholesky(self._gmm, lower=True)
