#!/usr/bin/env python

import numpy as np
import math
import copy

from floripy.models.ellipsoid import Ellipsoid
from floripy.file_formats import yamlio
from floripy.mathutils import xform as tr
from floripy.mathutils import geometry as geom
from floripy.mathutils.linalg import unitized


#------------------------------------------------------------------------------

class EllipsoidTrajectory(Trajectory):
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

