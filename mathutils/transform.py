#!/usr/bin/env python
'''
This module contains functions to convert Euler angles to Euler parameters and
vice-versa. The Euler parameters are defined thus:
    ep0 = cos(theta/2)
    ep1 = n_1*sin(theta/2)
    ep2 = n_2*sin(theta/2)
    ep3 = n_3*sin(theta/2)
    where n_1, n_2, and n_3 are the three components of an unit vector.
    Look at https://pyrr.readthedocs.org/en/latest/index.html if useful.
All angles are in radians.

'''

import math
import numpy as np
from numpy import linalg as la
from numpy import random as rn

#-------------------------------------------------
def perm_tensor():
    '''
    This function returns the permutation tensor.
    '''
    epsilon = np.zeros((3,3,3))
    epsilon[1,2,0] =  1.0
    epsilon[2,1,0] = -1.0
    epsilon[0,2,1] = -1.0
    epsilon[2,0,1] =  1.0
    epsilon[0,1,2] =  1.0
    epsilon[1,0,2] = -1.0
    return epsilon

#-------------------------------------------------

def generate_eulerparam():
    '''
    Generates a random set of Euler parameters.
    '''
    axis, angle = generate_axisangle()
    ep = axisangle2eulerparam(axis, angle)
    return ep

#-------------------------------------------------

def generate_axisangle():
    '''
    Generates a random pair of axis-angle. The axis ia a random vector from
    the surface of a unit sphere. Algorithm from Allen & Tildesley p. 349.
    Angle is a random number from [0.0, 2*pi).
    '''
    axis = np.zeros((3,))
    angle = 2.0*np.pi*rn.random()
    while True:
        zeta1 = 2.0*rn.random() - 1.0
        zeta2 = 2.0*rn.random() - 1.0
        zetasq = zeta1**2 + zeta2**2
        if zetasq <= 1.0:
            break
    rt = np.sqrt(1.0-zetasq)
    axis[0] = 2.0*zeta1*rt
    axis[1] = 2.0*zeta2*rt
    axis[2] = 1.0 - 2.0*zetasq
    return (axis, angle)


#-------------------------------------------------

def eulerparam2eulerangle(ep):
    '''
    This function converts Euler Parameters to Euler Angles.
    Euler Angles represent 3-1-3 (z-x-z) rotation.
    See Eq(2.143) of DMS.
    '''
    ep23 = ep[2]*ep[3]
    ep01 = ep[0]*ep[1]

    theta = np.arccos( 2.0*(ep[0]**2 +  ep[3]**2) - 1.0 )
    sin_theta = np.sin(theta)
    phi = np.arccos( -2.0*(ep23 - ep01)/sin_theta )
    psi = np.arccos(  2.0*(ep23 + ep01)/sin_theta )
    ea = np.asarray( (phi, theta, psi) )
    return ea

#-------------------------------------------------

def axisangle2eulerparam(axis, angle):
    '''
    Converts axis angle to Euler Parameters.
    Ref. Eq(2.10) of DMS.
    '''
    ep = np.empty(4)

    unit_axis = axis/la.norm(axis)

    sin_half_angle = np.sin(0.5*angle)
    cos_half_angle = np.cos(0.5*angle)

    ep[0] = cos_half_angle
    ep[1] = unit_axis[0] * sin_half_angle
    ep[2] = unit_axis[1] * sin_half_angle
    ep[3] = unit_axis[2] * sin_half_angle

    return ep

#-------------------------------------------------

def rotation_matrix2eulerangle(rotmat, order, body_fixed=True):
    '''
    Ref: Dmitry Savransky and N. Jeremy Kasdin, An Efficient Method for
        Extracting Euler Angles from Direction Cosine Matrices,
        http://spaceisbig.net/docs/findEulerAngs.pdf

    '''
    A = np.copy(rotmat)
    ea = np.zeros((3,))
    rotset = np.array([int(x) for x in order], dtype='i4')
    n = np.unique(rotset).size
    ax2neginds = [7, 2, 3]
    i = rotset[0] - 1
    j = rotset[1] - 1

    if n == 3:
        A = A*np.array([[1, -1, 1], [1, 1,-1], [-1, 1, 1]])
        if not body_fixed:
            A = A.T

        k = rotset[2] - 1
        c2 = math.hypot(A[i,i], A[i,j])

        ea[1] = math.atan2(A[i,k], c2)
        if c2 > np.finfo('f8').eps:
            ea[0] = math.atan2(A[j,k]/c2, A[k,k]/c2)
            ea[2] = math.atan2(A[i,j]/c2, A[i,i]/c2)
        else:
            ea[0] = 0.0
            ea[2] = 0.0

    elif n == 2:
        A[ax2neginds[j]] = -A[ax2neginds[j]]
        if not body_fixed:
            A = A.T

        p = 4 - (i + j)
        s2 = math.hypot(A[i,p], A[i,j])

        ea[0] = math.atan2(A[j,i]/s2, A[p,i]/s2)
        ea[1] = math.atan2(s2, A[i,i])
        ea[2] = math.atan2(A[i,j]/s2, A[i,p]/s2)
    else:
        raise ValueError('Illegal value {0} of n'.format(n))
    return ea

#-------------------------------------------------
def rotation_matrix2eulerparam(rotmat):
    '''
    Converts rotation matrix to Euler parameters.
    http://www.gamasutra.com/view/feature/131686/rotating_objects_using_quaternions.php
    '''
    trace = np.trace(rotmat)
    nxt = [1, 2, 0]
    q = np.zeros((4,))
    if trace > 0.0 :
        s = np.sqrt(trace + 1)
        w = s/2
        s = 1.0/(2*s)
        x = (rotmat[1,2] - rotmat[2,1])*s
        y = (rotmat[2,0] - rotmat[0,2])*s
        z = (rotmat[0,1] - rotmat[1,0])*s
        ep = np.array([w, x, y, z])
    else:
        i = 0
        if rotmat[1,1] > rotmat[0,0] :
            i = 1
        if rotmat[2,2] > rotmat[i,i] :
            i = 2
        j = nxt[i]
        k = nxt[j]
        s = np.sqrt((rotmat[i,i] - (rotmat[j,j] + rotmat[k,k])) + 1.0)
        q[i] = s/2
        if s != 0.0:
            s = 0.5/s
        q[3] = (rotmat[j,k] - rotmat[k,j])*s
        q[j] = (rotmat[i,j] + rotmat[j,i])*s
        q[k] = (rotmat[i,k] + rotmat[k,i])*s
        ep = np.array([q[3], q[0], q[1], q[2]])
    return ep


#-------------------------------------------------

def eulerparam2axisangle(ep):
    '''
    Converts Euler Parameters to axis angle.
    Ref. Invert Eq(2.10) of DMS.
    '''
    axis = np.empty(3)

    half_theta = np.arccos(ep[0])
    theta = 2.0*half_theta
    sin_half_theta = np.sin(half_theta)

    axis[0] = ep[1]/sin_half_theta
    axis[1] = ep[2]/sin_half_theta
    axis[2] = ep[3]/sin_half_theta

    return (axis, theta)

#-------------------------------------------------
def eulerangle2eulerparam(ea, order='313', fixed='body'):
    '''
    This function converts Euler Angles (3-1-3 or z-x-z rotation) to Euler
    parameters. See Eq(2.142) of DMS.
    '''
    if fixed == 'space':
        phi   = ea[2]
        theta = ea[1]
        psi   = ea[0]
    else:
        phi   = ea[0]
        theta = ea[1]
        psi   = ea[2]

    ep = np.empty(4)

    sin_theta_half = np.sin(theta/2.0)
    cos_theta_half = np.cos(theta/2.0)
    sin_phi_plus_psi = np.sin( (phi+psi)/2.0 )
    cos_phi_plus_psi = np.cos( (phi+psi)/2.0 )
    sin_phi_minus_psi = np.sin( (phi-psi)/2.0 )
    cos_phi_minus_psi = np.cos( (phi-psi)/2.0 )

    ep[0] = cos_theta_half * cos_phi_plus_psi
    ep[1] = sin_theta_half * cos_phi_minus_psi
    ep[2] = sin_theta_half * sin_phi_minus_psi
    ep[3] = cos_theta_half * sin_phi_plus_psi

    return ep

#-------------------------------------------------

def any2eulerparam(orientation):
    '''
    orientation is a dict.
    '''
    if orientation['repr'] == 'eulerparam':
        pass
    elif orientation['repr'] == 'rotmat':
        rotmat = np.asarray(orientation['rotmat']) #Alias rotation matrix
        orientation['ep'] = rotation_matrix2eulerparam(rotmat)
        orientation['repr'] = 'eulerparam'
        orientation.pop('rotmat', None)
    elif orientation['repr'] == 'axisangle':
        axis = orientation['axis']
        if orientation['angleunit'] == 'degree':
            angle = np.deg2rad(orientation['angle'])
        elif orientation['angleunit'] == 'radian':
            angle = orientation['angle']
        else:
            raise ValueError('Unknown angleunit ' + orientation['angleunit'])
        orientation['ep'] = axisangle2eulerparam(axis, angle)
        orientation['repr'] = 'eulerparam'
        for key in ['axis', 'angle']:
            orientation.pop(key, None)
    elif orientation['repr'] == 'eulerangle':
        if orientation['angleunit'] == 'degree':
            angles = np.deg2rad(orientation['angles'])
        elif orientation['angleunit'] == 'radian':
            angles = orientation['angles']
        else:
            raise ValueError('Unknown angleunit ' + orientation['angleunit'])
        order = orientation['order']
        fixed = orientation['fixed']
        orientation['ep'] = eulerangle2eulerparam(angles, order, fixed)
        orientation['repr'] = 'eulerparam'
        for key in ['angles', 'order', 'fixed']:
            orientation.pop(key, None)
    else:
        raise ValueError(
                'Unknown representation{0}\n'.format(orientation['repr']))
    return orientation

#-------------------------------------------------

def shifter_decomp_E(ep):
    '''
    Create the shifter decomposition matrices from the Euler parameters. These
    matrices are denoted by E and Ebar. See Eq(2.22) and (2.23) of DMS.
    '''
    E = np.zeros((3,4))

    E[0,0] = -ep[1]; E[0,1] =  ep[0]; E[0,2] = -ep[3]; E[0,3] =  ep[2]
    E[1,0] = -ep[2]; E[1,1] =  ep[3]; E[1,2] =  ep[0]; E[1,3] = -ep[1]
    E[2,0] = -ep[3]; E[2,1] = -ep[2]; E[2,2] =  ep[1]; E[2,3] =  ep[0]

    return E
#-------------------------------------------------
def shifter_decomp_Ebar(ep):
    '''
    Create the shifter decomposition matrices from the Euler parameters. These
    matrices are denoted by E and Ebar. See Eq(2.22) and (2.23) of DMS.
    '''
    Ebar = np.zeros((3,4))

    Ebar[0,0] = -ep[1]; Ebar[0,1] =  ep[0]; Ebar[0,2] =  ep[3]; Ebar[0,3] = -ep[2]
    Ebar[1,0] = -ep[2]; Ebar[1,1] = -ep[3]; Ebar[1,2] =  ep[0]; Ebar[1,3] =  ep[1]
    Ebar[2,0] = -ep[3]; Ebar[2,1] =  ep[2]; Ebar[2,2] = -ep[1]; Ebar[2,3] =  ep[0]

    return Ebar
#-------------------------------------------------
def shifter(ep):
    '''
    Create the shifter matrix from the given Euler parameters.
    See Eq(2.14) of DMS.
    '''
    ep0sq = ep[0]**2
    ep1sq = ep[1]**2
    ep2sq = ep[2]**2
    ep3sq = ep[3]**2
    ep0ep1 = ep[0]*ep[1]
    ep0ep2 = ep[0]*ep[2]
    ep0ep3 = ep[0]*ep[3]
    ep1ep2 = ep[1]*ep[2]
    ep1ep3 = ep[1]*ep[3]
    ep2ep3 = ep[2]*ep[3]

    S = np.empty((3,3))

    S[0,0] = 2.0*(ep0sq + ep1sq) - 1.0
    S[0,1] = 2.0*(ep1ep2 - ep0ep3)
    S[0,2] = 2.0*(ep1ep3 + ep0ep2)
    S[1,0] = 2.0*(ep1ep2 + ep0ep3)
    S[1,1] = 2.0*(ep0sq + ep2sq) - 1.0
    S[1,2] = 2.0*(ep2ep3 - ep0ep1)
    S[2,0] = 2.0*(ep1ep3 - ep0ep2)
    S[2,1] = 2.0*(ep2ep3 + ep0ep1)
    S[2,2] = 2.0*(ep0sq + ep3sq) - 1.0

    return S

#-------------------------------------------------
def rotation_matrix(axis, angle):
    '''
    Returns the matrix describing the rotation about axis <axis> by
    angle <angle>.
    '''
    R = np.zeros((3,3))
    sin = np.sin(angle)
    cos = np.cos(angle)
    icos = 1.0 - cos
    R[0,0] = axis[0]*axis[0]*icos + cos
    R[0,1] = axis[0]*axis[1]*icos - axis[2]*sin
    R[0,2] = axis[0]*axis[2]*icos + axis[1]*sin
    R[1,0] = axis[0]*axis[1]*icos + axis[2]*sin
    R[1,1] = axis[1]*axis[1]*icos + cos
    R[1,2] = axis[1]*axis[2]*icos - axis[0]*sin
    R[2,0] = axis[2]*axis[0]*icos - axis[1]*sin
    R[2,1] = axis[1]*axis[2]*icos + axis[0]*sin
    R[2,2] = axis[2]*axis[2]*icos + cos
    return R

#-------------------------------------------------
def epdot2ang_vel_transform_mat(ep):
    '''
    Returns the matrix that transforms epdot to angular velocity vector.
    Ref: Eq(2.58) of DMS.
    '''
    E = shifter_decomp_E(ep)
    return 2.0*E

#-------------------------------------------------
def epdot2ang_vel(ep, epdot):
    '''
    Returns the angular velocity vector from epdot.
    Ref: Eq(2.58) of DMS.
    '''
    mat = epdot2ang_vel_transform_mat(ep)
    ang_vel = np.dot(mat, epdot)
    return ang_vel

#-------------------------------------------------
def ang_vel2epdot_transform_mat(ep):
    '''
    Returns the matrix that transforms epdot to angular velocity vector.
    Ref: Invert Eq(2.58) of DMS, using the fact that the sum of the squares of
    the Euler parameters is unity, and the resulting 4x4 matrix will be
    orthogonal. Hence the inverse is its transpose. Now throw away the last
    column as it multiples the 4th component of ang_vel (= 0). The factor of
    half appears as there is a factor of 2 in the pre-inverted equation.
    '''
    mat = 0.5*shifter_decomp_E(ep)
    return mat.T

#-------------------------------------------------
def ang_vel2epdot(ep, ang_vel):
    '''
    Returns epdot from the angular velocity vector.
    Ref: Invert Eq(2.58) of DMS. See the note in the docstring of
    the function ang_vel2epdot_transform_mat.
    '''
    mat = ang_vel2epdot_transform_mat(ep)
    epdot = np.dot(mat, ang_vel)
    return epdot

#-------------------------------------------------
def ang_vel_mat(ang_vel):
    '''
    Depending on whether omega is global or local, Omega will be the global or
    local angular velocity matrix.
    '''
    Omega = np.zeros((3,3))
    Omega[0,1] = -ang_vel[2]
    Omega[0,2] =  ang_vel[1]
    Omega[1,2] = -ang_vel[0]
    Omega = Omega - Omega.T
    return Omega

#-------------------------------------------------
def shifter_dot(S, ang_vel):
    '''
    S: shifter
    Angular velocity can be obtained from epdot if desired.
    '''
    Omega = ang_vel_mat(ang_vel)
    SDot = np.dot(Omega, S)
    return SDot

#----------------------------------------------------------------------
def cross_mat(r):
    mat = np.zeros((3,3))
    mat[0,1] = -r[2]
    mat[0,2] = r[1]
    mat[1,0] = r[2]
    mat[1,2] = -r[0]
    mat[2,0] = -r[1]
    mat[2,1] = r[0]
    return mat

#----------------------------------------------------------------------

def dcm(B, A):
    '''
    C_ij = a_i . b_j. Gives the orientation of B frame w.r.t A frame.
    A and B are (3,3) matrices whose rows give the corresponding coordinate
    axis.
    '''
    return np.dot(A, B.T)

#----------------------------------------------------------------------

def dcm_to_euler_angles(dm, seq=[1, 2, 3], space=False):
    a11 = dm[0,0]; a12 = dm[0,1]; a13 = dm[0,2]
    a21 = dm[1,0]; a22 = dm[1,1]; a23 = dm[1,2]
    a31 = dm[2,0]; a32 = dm[2,1]; a33 = dm[2,2]
    try:
        theta_1 = math.atan(-a32/a33)
    except ZeroDivisionError:
        theta_1 = 0.0
    sin_theta_1 = math.sin(theta_1)
    cos_theta_1 = math.cos(theta_1)
    sin_theta_3 = a13*sin_theta_1 + a12*cos_theta_1
    cos_theta_3 = a23*sin_theta_1 + a22*cos_theta_1
    theta_3 = math.atan2(sin_theta_3, cos_theta_3)
    sin_theta_2 = a31
    cos_theta_2 = -a21*sin_theta_3 + a11*cos_theta_3
    theta_2 = math.atan2(sin_theta_2, cos_theta_2)
    return np.array([theta_1, theta_2, theta_3])

#----------------------------------------------------------------------

def main():
    print( 'Running transform.main()...' )
    print( '===========================' )
    print( '1. Permutation tensor' )
    print( perm_tensor() )

    print( '2. Generating skew matrix with the permutation tensor and a vector' )
    print( '---------------------------------------------------------------' )
    x = np.array([1, 2, 3])
    print( 'Vector:', x )
    print( 'Skew matrix')
    print( np.dot(perm_tensor(), x) )

    print( '3. Generate Euler parameter' )
    print( '---------------------------' )
    ep = generate_eulerparam()
    print('Euler parameters', ep)
    print( 'Sum of the squares of Euler parameters:', np.dot(ep,ep) )

    print( '4. Euler parameter to Euler angles and back' )
    print( '----------------------------------' )
    print( 'Euler parameters:', ep )
    ea = eulerparam2eulerangle( ep )
    print( 'Euler angles(deg):', np.rad2deg(ea) )
    ep = eulerangle2eulerparam( ea )
    print( 'Euler parameters:', ep )

    print( '5. Euler parameter to axis angle and back' )
    print( '-----------------------------------------' )
    print( 'Euler parameters:', ep )
    axis, angle = eulerparam2axisangle( ep )
    print( 'Axis:', axis, 'angle(deg):', np.rad2deg(angle) )
    ep = axisangle2eulerparam(axis, angle)
    print( 'Euler parameters:', ep )


#-----------------------------------------------------

if __name__ == '__main__':
    main()
