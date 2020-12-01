#!/usr/bin/env python
'''
Classes for various joint models.
Ref. Roy Featherstone, Rigid Body Dynamics, Table 4.1, p.79
'''
import math
import numpy as np
from abc import ABCMeta, abstractmethod, abstractproperty

from floripy.mathutils import xform as tr
from floripy.mathutils.linalg import cross_mat, unitized


class JointBase:
    __metaclass__ = ABCMeta
    def __init__(self, category, pf, sf, xi, r, zeta, prox_shifter,
            dist_shifter, mss, css, num_gencoord, num_genspeed):
        #The motion subspace and the constraint subspace are required to be
        #(6 x n) ndarrays.
        assert mss.ndim == 2
        self._category = category
        self._pf = pf
        self._sf = sf
        self._xi = xi
        self._r = r
        self._zeta = zeta
        self._prox_shifter = prox_shifter
        self._dist_shifter = dist_shifter
        self._mss = mss #Motion subspace
        self._css = css #Constraint subspace
        self._num_gencoord = num_gencoord
        self._num_genspeed = num_genspeed
        self._gencoord = np.zeros((num_gencoord,))
        #joint_shifter shifts from distal frame to proximal frame.
        self._joint_shifter = np.zeros((3,3))
        self._shifter66 = np.eye(6)
        self._matH = np.zeros((6,6))
        self._jpvm = np.zeros((6,num_genspeed))


    @abstractmethod
    def get_gencoord_dot(self, u):
        pass

    def get_gencoord(self):
        return self._gencoord

    def get_jpvm(self):
        '''Returns the joint partial velocity matrix.
        '''
        return self._jpvm

    def get_mss(self):
        '''Returns the joint motion subspace.

        '''
        return self._mss

    def get_css(self):
        '''Returns the joint constraint subspace.

        '''
        return self._css

    def get_shifter66(self):
        '''Return the 6x6 transform in world.

        '''
        return self._shifter66


    @property
    def num_genspeed(self):
        return self._num_genspeed

    @property
    def num_gencoord(self):
        return self._num_gencoord

    @property
    def num_constraints(self):
        return self._css.shape[1]

    @property
    def predecessor(self):
        return self._pf.bid

    @property
    def successor(self):
        return self._sf.bid

    @property
    def connects(self):
        return (self._pf.bid, self._sf.bid)


#------------------------------------------------------------------------------

class Revolute(JointBase):
    def __init__(self, **kwargs):
        num_gencoord = 1
        num_genspeed = 1
        pf = kwargs['pf']
        sf = kwargs['sf']
        pf_edge = np.array(kwargs['pf_edge'])
        xi = np.mean(pf_edge, axis=0)
        sf_edge = np.array(kwargs['sf_edge'])
        zeta = np.mean(sf_edge, axis=0)
        r = np.zeros((3,))

        pf_zdir = np.array([0, 0, 1])
        prox_zdir = unitized(pf_edge[1,:]-pf_edge[0,:])
        angle = math.acos(np.dot(pf_zdir, prox_zdir))
        if abs(angle) > np.finfo(np.float64).eps:
            axis = np.cross(pf_zdir, prox_zdir)
            axis = unitized(axis)
            prox_shifter = tr.get_shiftmat_axis_angle(axis, angle, forward=False)
        else:
            prox_shifter = np.eye(3)

        sf_zdir = np.array([0, 0, 1])
        dist_zdir = unitized(sf_edge[1,:]-sf_edge[0,:])
        angle = math.acos(np.dot(sf_zdir, dist_zdir))
        if abs(angle) > np.finfo(np.float64).eps:
            axis = np.cross(sf_zdir, dist_zdir)
            axis = unitized(axis)
            dist_shifter = tr.get_shiftmat_axis_angle(axis, angle, forward=True)
        else:
            dist_shifter = np.eye(3)

        mss = np.zeros((6,1))
        mss[2,0] = 1.0
        css = np.zeros((6,5))
        css[0,0] = 1.0
        css[1,1] = 1.0
        css[3,2] = 1.0
        css[4,3] = 1.0
        css[5,4] = 1.0

        super(Revolute, self).__init__(kwargs['category'], pf, sf, xi, r, zeta,
                prox_shifter, dist_shifter, mss, css, num_gencoord, num_genspeed)

        if self._category == 'tree':
            angle = kwargs['angle']
            self.update(np.array([angle]))

 
    def update(self, q):
        self._gencoord[:] = q

        js = tr.get_shiftmat_axis_angle(self._mss[0:3,0], self._gencoord[0],
                forward=False)

        pfo = self._pf.get_com()
        pfs = self._pf.get_shifter()
        prfs = np.dot(pfs, self._prox_shifter)
        sfs = np.einsum('ij,jk,kl', prfs, js, self._dist_shifter)
        mu = np.dot(sfs, self._zeta)
        nu = np.dot(pfs, self._xi+np.dot(self._prox_shifter, self._r))
        mucross = cross_mat(mu)
        nucross = cross_mat(nu)
        sfo = pfo + nu - mu
        self._shifter66[3:6,0:3] = mucross - nucross
        self._matH[0:3,0:3] = prfs
        self._matH[3:6,0:3] = np.dot(mucross, prfs)
        self._matH[3:6,3:6] = prfs
        self._jpvm = np.dot(self._matH, self._mss)
        self._sf.update_position(sfo, sfs)


    def selfupdate(self):
        q = np.copy(self._gencoord)
        self.update(q)


    def renormalize(self):
        pass


    def get_gencoord_dot(self, u):
        return u


    def get_constraint_mat(self):
        pfs = self._pf.get_shifter()
        sfs = self._sf.get_shifter()
        prfs = np.dot(pfs, self._prox_shifter)
        mu = np.dot(sfs, self._zeta)
        nu = np.dot(pfs, self._xi)
        mucross = cross_mat(mu)
        nucross = cross_mat(nu)
        pfcm = np.zeros((6,6))
        pfcm[0:3,0:3] = prfs.T
        pfcm[3:6,0:3] = -np.dot(prfs.T, nucross)
        pfcm[3:6,3:6] = prfs.T
        sfcm = np.zeros((6,6))
        sfcm[0:3,0:3] = prfs.T
        sfcm[3:6,0:3] = -np.dot(prfs.T, mucross)
        sfcm[3:6,3:6] = prfs.T
        return np.dot(self._css.T, pfcm), np.dot(self._css.T, sfcm)


#------------------------------------------------------------------------------


class Free(JointBase):
    def __init__(self, **kwargs):
        num_gencoord = 7
        num_genspeed = 6
        xi = np.asarray(kwargs['from'])
        zeta = np.asarray(kwargs['to'])
        r = np.asarray(kwargs['r'])
        prox_shifter = np.eye(3)
        dist_shifter = np.eye(3)
        mss = np.eye(6)
        css = np.array([[],[]])

        super(Free, self).__init__(kwargs['category'], kwargs['pf'],
                kwargs['sf'], xi, r, zeta, prox_shifter, dist_shifter, mss, css,
                num_gencoord, num_genspeed)

        if self._category == 'tree':
            orientation = kwargs['orientation']
            quat = tr.any_to_quat(orientation)
            self.update(np.concatenate((quat, self._r)))


    def update(self, q):
        self._gencoord[0:4] = tr.normalize_quat(q[0:4])
        self._gencoord[4:7] = q[4:7]
        self._r[0:3] = self._gencoord[4:7]

        js = tr.get_shiftmat_quat(self._gencoord[0:4], forward=False)

        pfo = self._pf.get_com()
        pfs = self._pf.get_shifter()
        prfs = np.dot(pfs, self._prox_shifter)
        sfs = np.einsum('ij,jk,kl', prfs, js, self._dist_shifter)
        mu = np.dot(sfs, self._zeta)
        nu = np.dot(pfs, self._xi+np.dot(self._prox_shifter, self._r))
        mucross = cross_mat(mu)
        nucross = cross_mat(nu)
        sfo = pfo + nu - mu
        self._shifter66[3:6,0:3] = mucross - nucross
        self._matH[0:3,0:3] = prfs
        self._matH[3:6,0:3] = np.dot(mucross, prfs)
        self._matH[3:6,3:6] = prfs
        self._jpvm = np.dot(self._matH, self._mss)
        self._sf.update_position(sfo, sfs)


    def selfupdate(self):
        q = np.copy(self._gencoord)
        self.update(q)


    def get_gencoord_dot(self, u):
        qdot = np.zeros((self.num_gencoord,))
        qdot[0:4] = tr.ang_vel_to_quat_deriv(self._gencoord[0:4], u[0:3])
        qdot[4:7] = u[3:6]
        return qdot


    def renormalize(self):
#       print(self._gencoord[0:4])
        tr.normalize_quat(self._gencoord[0:4])
#       print(self._gencoord[0:4])


    def get_constraint_mat(self):
        raise NotImplementedError()

#------------------------------------------------------------------------------

class Spherical(JointBase):
    def __init__(self, **kwargs):
        num_gencoord = 4
        num_genspeed = 3
        xi = np.asarray(kwargs['from'])
        zeta = np.asarray(kwargs['to'])
        r = np.zeros((3,))
        prox_shifter = np.eye(3)
        dist_shifter = np.eye(3)
        mss = np.zeros((6,3))
        mss[0,0] = 1.0
        mss[1,1] = 1.0
        mss[2,2] = 1.0
        css = np.zeros((6,3))
        css[3,0] = 1.0
        css[4,1] = 1.0
        css[5,2] = 1.0

        super(Spherical, self).__init__(kwargs['category'], kwargs['pf'],
                kwargs['sf'], xi, r, zeta, prox_shifter, dist_shifter, mss, css,
                num_gencoord, num_genspeed)
        
        if self._category == 'tree':
            orientation = kwargs['orientation']
            quat = tr.any_to_quat(orientation)
            self.update(quat)


    def update(self, q):
        self._gencoord[:] = tr.normalize_quat(q)

        js = tr.get_shiftmat_quat(self._gencoord, forward=False)

        pfo = self._pf.get_com()
        pfs = self._pf.get_shifter()
        prfs = np.dot(pfs, self._prox_shifter)
        sfs = np.einsum('ij,jk,kl', prfs, js, self._dist_shifter)
        mu = np.dot(sfs, self._zeta)
        nu = np.dot(pfs, self._xi+np.dot(self._prox_shifter,self._r))
        mucross = cross_mat(mu)
        nucross = cross_mat(nu)
        sfo = pfo + nu - mu
        self._shifter66[3:6,0:3] = mucross - nucross
        self._matH[0:3,0:3] = prfs
        self._matH[3:6,0:3] = np.dot(mucross, prfs)
        self._matH[3:6,3:6] = prfs
        self._jpvm = np.dot(self._matH, self._mss)
        self._sf.update_position(sfo, sfs)


    def selfupdate(self):
        q = np.copy(self._gencoord)
        self.update(q)


    def get_gencoord_dot(self, u):
        qdot = tr.ang_vel_to_quat_deriv(self._gencoord, u)
        return qdot


    def get_constraint_mat(self):
        pfs = self._pf.get_shifter()
        sfs = self._sf.get_shifter()
        prfs = np.dot(pfs, self._prox_shifter)
        mu = np.dot(sfs, self._zeta)
        nu = np.dot(pfs, self._xi)
        mucross = cross_mat(mu)
        nucross = cross_mat(nu)
        pfcm = np.zeros((6,6))
        pfcm[0:3,0:3] = prfs.T
        pfcm[3:6,0:3] = -np.dot(prfs.T, nucross)
        pfcm[3:6,3:6] = prfs.T
        sfcm = np.zeros((6,6))
        sfcm[0:3,0:3] = prfs.T
        sfcm[3:6,0:3] = -np.dot(prfs.T, mucross)
        sfcm[3:6,3:6] = prfs.T
        return np.dot(self._css.T, pfcm), np.dot(self._css.T, sfcm)

#------------------------------------------------------------------------------

class Root(Free):
    def __init__(self, **kwargs):
        self._rooted = True
        kwargs['category'] = 'tree'
        kwargs['from'] = np.zeros((3,))
        kwargs['r'] = np.zeros((3,))
        super(Root, self).__init__(**kwargs)


    def update(self, q):
        self._gencoord[0:4] = tr.normalize_quat(q[0:4])
        if not self._rooted:
            self._gencoord[4:7] = q[4:7]
            self._r[0:3] = self._gencoord[4:7]

        js = tr.get_shiftmat_quat(self._gencoord[0:4], forward=False)

        pfo = self._pf.get_com()
        pfs = self._pf.get_shifter()
        prfs = np.dot(pfs, self._prox_shifter)
        sfs = np.einsum('ij,jk,kl', prfs, js, self._dist_shifter)
        mu = np.dot(sfs, self._zeta)
        nu = np.dot(pfs, self._xi+np.dot(self._prox_shifter,self._r))
        mucross = cross_mat(mu)
        nucross = cross_mat(nu)
        sfo = pfo + nu - mu
        self._shifter66[3:6,0:3] = mucross - nucross
        self._matH[0:3,0:3] = prfs
        self._matH[3:6,0:3] = np.dot(mucross, prfs)
        self._matH[3:6,3:6] = prfs
        self._jpvm = np.dot(self._matH, self._mss)
        self._sf.update_position(sfo, sfs)


    def selfupdate(self):
        q = np.copy(self._gencoord)
        self.update(q)


#   def get_constraint_mat(self):
#       raise NotImplementedError()

    def uproot(self):
        self._rooted = False


    def reroot(self, root_sphere_radius=0.0, loc=[0.0, 0.0, 0.0]):
        r = np.copy(self._gencoord[4:7])
        delta_r = np.linalg.norm(r)
        if delta_r > root_sphere_radius:
            if self.is_rooted():
                self.uproot()
                uprooted_here = True
            else:
                uprooted_here = False
            q = self._gencoord[:]
            q[4:7] = np.asarray(loc)
            self.update(q)
            if uprooted_here:
                self._rooted = True
            return r
        else:
            return np.zeros((3,))


    def is_rooted(self):
        return self._rooted



#------------------------------------------------------------------------------

class Fixed(JointBase):
    def __init__(self, **kwargs):
        num_gencoord = 0
        num_genspeed = 0
        xi = np.asarray(kwargs['from'])
        zeta = np.asarray(kwargs['to'])
        r = np.zeros((3,))
        prox_shifter = np.eye(3)
        dist_shifter = np.eye(3)
        mss = None
        css = np.eye(6)

        super(Fixed, self).__init__(kwargs['category'], kwargs['pf'],
                kwargs['sf'], xi, r, zeta, prox_shifter, dist_shifter, mss, css,
                num_gencoord, num_genspeed)


    def update(self, q):
        self._gencoord[:] = q

        js = np.eye(3)

        pfo = self._pf.get_com()
        pfs = self._pf.get_shifter()
        prfs = np.dot(pfs, self._prox_shifter)
        sfs = np.einsum('ij,jk,kl', prfs, js, self._dist_shifter)
        mu = np.dot(sfs, self._zeta)
        nu = np.dot(pfs, self._xi+np.dot(self._prox_shifter,self._r))
        mucross = cross_mat(mu)
        nucross = cross_mat(nu)
        sfo = pfo + nu - mu
        self._shifter66[3:6,0:3] = mucross - nucross
        self._matH[0:3,0:3] = prfs
        self._matH[3:6,0:3] = np.dot(mucross, prfs)
        self._matH[3:6,3:6] = prfs
        self._jpvm = np.dot(self._matH, self._mss)
        self._sf.update_position(sfo, sfs)


    def selfupdate(self):
        q = np.copy(self._gencoord)
        self.update(q)


    def get_gencoord_dot(self, u):
        return None


    def get_constraint_mat(self):
        raise NotImplementedError

#------------------------------------------------------------------------------

def create(joint_type, **kwargs):
    classes = {'root': Root, 'fixed': Fixed, 'revolute': Revolute,
            'spherical': Spherical, 'free': Free}
    return classes[joint_type](**kwargs)
