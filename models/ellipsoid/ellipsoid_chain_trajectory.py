#!/usr/bin/env python

import numpy as np
import math
import copy

from floripy.models.trajectory import Trajectory
from floripy.models.mbs_trajectory import MbsTrajectory
from floripy.models.modelbase import ModelBase
from floripy.file_formats import yamlio
from floripy.mathutils import xform as tr
from floripy.mathutils import geometry as geom
from floripy.mathutils.linalg import unitized


class EllipsoidChain(ModelBase):
    def __init__(self, a, b, c, el_coms, el_oris, joints):
        '''
        Parameters
        ----------
        a : float/ndarray
            Ellipsoid a
        b : float/ndarray
            Ellipsoid b
        c : float/ndarray
            Ellipsoid c
        el_coms : (N,3) ndarray, dtype float
            Ellipsoid c.o.m. in world frame.
        el_oris : (N,4) ndarray, dtype float
            Ellipsoid orientations in world frame.
        joints : dict
            The values of the joint variables will be overwritten.


        Returns
        -------
        An Ellipsoid_chain instance

        '''
        self.a = a
        self.b = b
        self.c = c
        self.el_coms = el_coms
        self.el_oris = el_oris
        self._joints = joints

        self.nel = self.el_coms.shape[0] #ellipsoids
        self.com = np.mean(self.el_coms, axis=0)
        #Root joint
        q = self.el_oris[0,:]
        to = tr.shift_vector_quat(-self.el_coms[0,:], q, forward=True)
        self._joints[1] = {'joint_type': 'root',
                            'to': to,
                            'orientation': {'repr': 'quat', 'quat': q}
                            }
        #Joint variables for non-root joints
        for jid in range(2,self.nel+1):
            #Joint `jid` connects body `jid-1` with body `jid`. The array
            #indices for accessing com and ori for body `k` is `k-1`.
            inv = tr.get_inverted_quat(self.el_oris[jid-2,:])
            quat = tr.get_quat_prod(inv, self.el_oris[jid-1,:])
            self._joints[jid]['orientation'] = {'repr': 'quat', 'quat': quat}


    @classmethod
    def create(cls, a, b, c, nel, ori):
        '''
        Creates a chain of ellipsoids
        ori : (nel, 4) ndarray
            Orientations of all ellipsoids
        '''
        el_oris = ori
        el_coms = np.zeros((nel, 3))
        joints = {}

        for k in range(1, nel):
            xi = np.array([a, 0, 0])
            zeta = np.array([-a, 0, 0])
            nu = tr.shift_vector_quat(xi, el_oris[k-1,:], forward=False)
            mu = tr.shift_vector_quat(zeta, el_oris[k,:], forward=False)
            el_coms[k,:] = el_coms[k-1,:] + nu - mu
            joints[k+1] = {'from': xi,
                         'to': zeta,
                         'pf': k,
                         'sf': k+1,
                        }

        ec = cls(a, b, c, el_coms, el_oris, joints)
        return ec


    @classmethod
    def from_df_mbs(cls, df_mbs, modelspec):
        '''
        Creates an EllipsoidChain instance from mbsdata and modeldata. The c.o.m.
        position and orientation is as calculated from mbsdata.

        Parameters
        ----------
        df_mbs : numpy structured array 
            Numpy structured array containing the fields com and shifter. Other
            fields, if present, are ignored.
        modelspec : dict

        Returns
        -------
        ec : An instance of EllipsoidChain

        '''
        bodies = modelspec['bodies']
        nb = len(bodies)
        #Parameters a, b, and c for the ellipsoids. These are same for all
        #ellipsoids, so just extract from body 1.
        a = bodies[1]['a']
        b = bodies[1]['b']
        c = bodies[1]['c']

        #joints dict from modelspec. The joints variables will be recalculated
        #inside __init__.
        joints = modelspec['joints']
        el_coms = df_mbs['com']
        el_oris = df_mbs['ori']
        ec = cls(a, b, c, el_coms, el_oris, joints)
        return ec


    @classmethod
    def from_df_traj(cls, df_traj, modelspec):
        bodies = modelspec['bodies']
        nb = len(bodies)

        #Parameters a, b, and c for the ellipsoids. These are same for all
        #ellipsoids, so just extract from body 1.
        a = bodies[1]['a']
        b = bodies[1]['b']
        c = bodies[1]['c']

        #joints dict from modelspec. The joints variables will be recalculated
        #inside __init__.
        joints = modelspec['joints']
        el_coms = df_traj['com']
        el_oris = df_traj['ori']
        ec = cls(a, b, c, el_coms, el_oris, joints)
        return ec


    def to_df_traj(self, time):
        traj_dtype = np.dtype([('time', 'f8'), ('com', 'f8', (self.nel,3)),
                                ('ori', 'f8', (self.nel,4))])
        df_traj = np.array([(time, self.el_coms, self.el_oris)],
                            dtype=traj_dtype)
        return df_traj


    def to_mbs(self, fn_yaml):
        #Create the bodies
        bodies = {}
        for k in range(self.nel):
            bodies[k+1] = {'ellipsoid': {'a': float(self.a),
                            'b': float(self.b), 'c': float(self.c)}}
        #Create the joints and convert all numpy types to lists or floats as
        #appropriate
        joints = copy.deepcopy(self._joints)
        #Root joint
        to = joints[1]['to']
        joints[1]['to'] = to.tolist()
        q = joints[1]['orientation']['quat']
        joints[1]['orientation']['quat'] = q.tolist()
        #Non-root joints (all spherical)
        for jid in range(2, self.nel+1):
            to = joints[jid]['to']
            joints[jid]['to'] = to.tolist()
            frm = joints[jid]['from']
            joints[jid]['from'] = frm.tolist()
            q = joints[jid]['orientation']['quat']
            joints[jid]['orientation']['quat'] = q.tolist()

        #Data for reconstructing the model from MBS data, None for an
        #EllipsoidChain instance
        internal = None
        #Write to yaml file
        yamlio.write({'bodies': bodies, 'joints': joints,
                        '{internal}': internal}, fn_yaml)


    def set_com(self, loc):
        delta = loc - self.com
        for k in range(self.nel):
            self.el_coms[k,:] += delta
        self.com = np.mean(self.el_coms, axis=0)
        #Update the root joint
        q = self.el_oris[0,:]
        to = tr.shift_vector_quat(-self.el_coms[0,:], q, forward=True)
        self._joints[1]['to'] = to


#------------------------------------------------------------------------------

class EllipsoidChainTrajectory(Trajectory):
    def __init__(self, fn_traj, fn_model):
        self._modelspec = yamlio.read(fn_model)
        self.nel = len(self._modelspec['bodies'])
        dtype = np.dtype([('time', 'f8'), ('com', 'f8', (self.nel,3)),
                        ('ori', 'f8', (self.nel,4))])
        super(EllipsoidChainTrajectory, self).__init__(fn_traj, dtype)


    def get_frame(self, i):
        frame = self._traj[i]
        ec = EllipsoidChain.from_df_traj(frame, self._modelspec)
        return (frame['time'], ec)


    @classmethod
    def append(cls, fn, fn_other):
        blocksize = 1024
        with open(fn, 'ab') as fh, open(fn_other, 'rb') as fh_other:
            while True:
                block = fh_other.read(blocksize)
                if block:
                    fh.write(block)
                else:
                    break


    def to_ascii(self, fn):
        with open(fn, 'w') as fh:
            nframes = len(self)
            for k in range(nframes):
                frame = self.get_frame(k)
                time = frame['time']
                com = frame['com']
                ori = frame['ori']
                fh.write('time: {0}\n'.format(time))
                fh.write('------------------------------\n')
                fh.write('com\n')
                for i in range(self.nel):
                    fh.write('  '.join([str(x) for x in com[i,:]])+'\n')
                fh.write('ori\n')
                for i in range(self.nel):
                    fh.write('  '.join([str(x) for x in ori[i,:]])+'\n')
                fh.write('*********************************************\n')


#-------------------------------------------------------------------------------

def modeltraj_from_mbstraj(fn_mbstraj, fn_modeltraj, fn_model):
    fh_traj = open(fn_modeltraj, 'wb')
    modelspec = yamlio.read(fn_model)
    mbstraj = MbsTrajectory(fn_mbstraj, fn_model)
    nframes = len(mbstraj)
    for k in range(nframes):
        df_mbs = mbstraj.get_frame(k)
        ec = EllipsoidChain.from_df_mbs(df_mbs, modelspec)
        df_traj = ec.to_df_traj()
        fh_traj.write(df_traj.tobytes())
    mbstraj.close()
    fh_traj.close()


def get_phi_theta(v):
    '''
    v: (3,) ndarray
    Returns phi and theta
    phi: Angle measured from the y-axis of the projection on the xy-plane
            0<=phi<=pi
    theta: Angle measured from the z-axis, 0<=theta<=pi
    '''
    vec = unitized(v)
    #vec projected on the x-y plane
    vec_proj_xy = unitized(np.array([vec[0], vec[1], 0]))
    ybv = np.array([0, 1, 0])
    zbv = np.array([0, 0, 1])
    cos_phi = np.dot(ybv, vec_proj_xy) 
    cos_theta = np.dot(zbv, vec) 

    if abs(cos_phi) > 1.0:
        if isclose(abs(cos_phi), 1.0, abs_tol=1e-12, rel_tol=1e-12):
            sign = math.copysign(1, cos_phi)
            cos_phi = sign
    phi = math.acos(cos_phi)
    phi_deg = math.degrees(phi)

    if abs(cos_theta) > 1.0:
        if isclose(abs(cos_theta), 1.0, abs_tol=1e-12, rel_tol=1e-12):
            sign = math.copysign(1, cos_theta)
            cos_theta = sign
    theta = math.acos(cos_theta)
    theta_deg = math.degrees(theta)
    return phi_deg, theta_deg

#-------------------------------------------------------------------------------

