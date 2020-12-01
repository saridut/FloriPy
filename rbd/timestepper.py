#!/usr/bin/env python
'''
Class implementing Kane's method.
'''
import math
import numpy as np
from numpy import linalg as la
from scipy import linalg as sla
from scipy.integrate import ode

class Timestepper(object):
    '''
    Interface to ODE integrator.
    '''
    def __init__(self, model, hydrodynamics, flowfield, **kwargs):
        np.set_printoptions(linewidth=160)
        self._model = model
        self._hydrodynamics = hydrodynamics
        self._flowfield = flowfield
        odeopts = kwargs['odeopts']
        self._time_start = kwargs['time_start']
        self._time_end = kwargs['time_end']
        self._time_stepsize = kwargs['time_stepsize']
        self._time_write_interval = kwargs['time_write_interval']
        self._time_pullback_interval = kwargs['time_pullback_interval']
        self._time_print_interval = kwargs['time_print_interval']
        self._time_flow_reversal = kwargs['time_flow_reversal']

        ngs = self._model.num_genspeed
        nc = self._model.num_constraints
#       print(ngs, nc)

        self._q = self._model.get_gencoord()
        self._wrk_mat = np.zeros((ngs, ngs))
        self._wrk_vec = np.zeros((ngs,))

        solver = ode(self._eval_rhs)
        solver.set_integrator(**odeopts)
        solver.set_initial_value(self._q, self._time_start)
        self._solver = solver


    def _eval_rhs(self, time, q):
        '''
        Evaluate the RHS for Kane's equations of motion.
        '''
        self._model.update(q, time)
        partial_mat = self._model.get_partial_mat()
        self._hydrodynamics.update(time)

        ngs = self._model.num_genspeed

        damping_mat = self._hydrodynamics.getrans_grm_ge(partial_mat)
        #Compute the LTL decomposition of damping_mat. damping_mat will be
        #overwritten with L.
        L = self.decomp_LTL(damping_mat, overwrite=True)
        LT = np.copy(L.T)

        b_vec = np.dot(partial_mat.T, self._hydrodynamics.grmvel())

        nc = self._model.num_constraints
        if nc > 0:
            #Solution using Cholesky decompostion of the grand resistance
            #matrix. The Lagrange multipliers are evaluated first.
            B = self._model.get_constraint_mat()
            Y = np.copy(B.T)
            z_vec = np.copy(b_vec)

            #Back substitution to solve LT*Y = BT and LT*z = b_vec
            for k in range(ngs-1, -1, -1):
                z_vec[k] = (z_vec[k] - np.dot(LT[k,k+1:], z_vec[k+1:]))/LT[k,k]
                Y[k,:] = (Y[k,:] - np.dot(LT[k,k+1:], Y[k+1:,:]))/LT[k,k]

            A = np.dot(Y.T, Y)
            w_vec = np.dot(Y.T, z_vec)

            #Least squares solution of the Lagrange multipliers
            lamda, residues, rank, sing_vals = sla.lstsq(A, w_vec,
                                overwrite_a=True, overwrite_b=True)
            #Update b_vec by subtracting BT*lamda
            b_vec -= np.dot(B.T, lamda)

        #Back substitution to solve LT*y = b_vec. The solution overwrites
        #b_vec.
        for k in range(ngs-1, -1, -1):
            b_vec[k] = (b_vec[k] - np.dot(b_vec[k+1:], LT[k,k+1:]))/LT[k,k]

        #Forward substitution to solve L*z = y. The solution overwrites the
        #right hand side.
        for k in range(ngs):
            b_vec[k] = (b_vec[k] - np.dot(b_vec[0:k], L[k,0:k]))/L[k,k]

        #Obtain qdot and return
        self._qdot = self._model.get_gencoord_dot(b_vec)
        return self._qdot


    @staticmethod
    def decomp_LTL(A, overwrite=False, zero_upper=True, out=None):
        '''
        Compute the LTL decomposition of a positive definite matrix.
        '''
        m, n = A.shape
        assert m == n
        if overwrite:
            L = A
        else:
            if out is not None:
                assert n,n == out.shape
                out[:,:] = 0.0
                L = out
            else:
                L = np.zeros((n,n))

        for i in range(n-1,-1,-1):
            L[i,i] = math.sqrt(A[i,i] - np.dot(L[i+1:,i],L[i+1:,i]))
            for j in range(i):
                L[i,j] = (A[i,j] - np.dot(L[i+1:,i], L[i+1:,j]))/L[i,i]

        if overwrite and zero_upper:
            for i in range(n):
                for j in range(i+1,n):
                    L[i,j] = 0.0
        return L


    def run(self):
        time = self._time_start
        print(time)
        time_elapsed = 0.0
        pullback_time_elapsed = 0.0
        print_time_elapsed = 0.0
#       flow_time_elapsed = 0.0
        step_dir = math.copysign(1, (self._time_end-self._time_start))

        while step_dir*(self._time_end-time) > 0 :
            time += step_dir*self._time_stepsize
            self._q = self._solver.integrate(time)
            self._model.update(self._q, time)
            if not self._solver.successful():
                print ('Integration failed; time = {0:g}'.format(time))
                self._model.close_files()
                raise SystemExit('Leaving...')
            else:
                #Update timers
                time_elapsed += self._time_stepsize
                pullback_time_elapsed += self._time_stepsize
                print_time_elapsed += self._time_stepsize
#               flow_time_elapsed += self._time_stepsize
                if print_time_elapsed >= self._time_print_interval:
                    print_time_elapsed = 0.0
                    print(time)
                if pullback_time_elapsed >= self._time_pullback_interval:
                    pullback_time_elapsed = 0.0
                    pulled_back = self._model.pullback()
                    if pulled_back:
                        self._q = self._model.get_gencoord()
                        self._solver.set_initial_value(self._q, time)
                #Fix unit quaternions
                #self._model.renormalize()
                #self._q = self._model.get_gencoord()
                #self._solver.set_initial_value(self._q, time)
                if time_elapsed >= self._time_write_interval:
                    time_elapsed = 0.0
                    self._model.tofile(time)

#               if flow_time_elapsed >= self._time_flow_reversal:
#                   sr = self._flowfield.get_property('strain_rate')
#                   self._flowfield.set_property('strain_rate', -sr)
#                   flow_time_elapsed = 0.0

        self._model.close_files()
