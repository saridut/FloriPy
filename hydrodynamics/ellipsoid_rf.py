#!/usr/bin/env python

import math
import mpmath
import numpy as np
from context import floripy
from floripy.mathutils.xform import *


#mpmath.mp.dps = 128
pi = math.pi


def get_force_torque_prolate(a, c, d, u_inf, omega_inf, E_inf):
    if c == 'zero':
        print('Reduction of prolate to a line not possible.')
        return float('nan'), float('nan')
    e = math.sqrt((a+c)*(a-c))/a
    e2 = e*e
    e3 = e2*e
    e5 = e3*e2
    one_m_e2 = 1.0 - e2
    one_p_e2 = 1.0 + e2
    two_m_e2 = 2.0 - e2
    two_e2_m_three = 2.0*e2 - 3.0
    three_m_e2 = 3.0 - e2
    three_e2_m_one = 3.0*e2 - 1.0
    three_m_five_e2 = 3.0 - 5.0*e2

    L = math.log1p(e) - math.log1p(-e)
    XA = (8.0/3)*e3/(-2*e + one_p_e2*L)
    YA = (16.0/3)*e3/(2*e + three_e2_m_one*L)
    XC = (4.0/3)*e3*one_m_e2/(2*e - one_m_e2*L)
    YC = (4.0/3)*e3*two_m_e2/(-2*e + one_p_e2*L)
    YH = (4.0/3)*e5/(-2*e + one_p_e2*L)

    epsilon = np.zeros((3,3,3))
    epsilon[1,2,0] =  1.0
    epsilon[2,1,0] = -1.0
    epsilon[0,2,1] = -1.0
    epsilon[2,0,1] =  1.0
    epsilon[0,1,2] =  1.0
    epsilon[1,0,2] = -1.0

    dd = np.outer(d,d)
    I = np.identity(3)
    A = 6*pi*a*(XA*dd + YA*(I-dd))
    force = np.dot(A, u_inf)
    C = 8*pi*(a**3)*(XC*dd + YC*(I-dd))
    torque = np.dot(C, omega_inf) - 8*pi*a**3*YH*np.einsum('ijl,l,k,jk',
                    epsilon, d, d, E_inf)
    return force, torque


def get_force_torque_oblate(a, c, d, u_inf, omega_inf, E_inf):
    if c == 'zero':
        XA = 8/(3*pi)
        YA = 16/(9*pi)
        XC = 4/(3*pi)
        YC = 4/(3*pi)
        YH = -4/(3*pi)
    else:
        e = math.sqrt((a+c)*(a-c))/a
        e2 = e*e
        e3 = e2*e
        e5 = e2*e3
        one_m_e2 = 1.0 - e2
        two_m_e2 = 2.0 - e2
        sqrt_one_m_e2 = math.sqrt(one_m_e2)
        C = math.atan(e/sqrt_one_m_e2)
        XA = (4.0/3)*e3/((2*e2-1)*C + e*sqrt_one_m_e2)
        YA = (8.0/3)*e3/((2*e2+1)*C - e*sqrt_one_m_e2)
        XC = (2.0/3)*e3/(C - e*sqrt_one_m_e2)
        YC = (2.0/3)*e3*two_m_e2/(e*sqrt_one_m_e2 - (1-2*e2)*C)
        YH = (-2.0/3)*e5/(e*sqrt_one_m_e2 - (1-2*e2)*C)

    epsilon = np.zeros((3,3,3))
    epsilon[1,2,0] =  1.0
    epsilon[2,1,0] = -1.0
    epsilon[0,2,1] = -1.0
    epsilon[2,0,1] =  1.0
    epsilon[0,1,2] =  1.0
    epsilon[1,0,2] = -1.0

    dd = np.outer(d,d)
    I = np.identity(3)
    K = 6*pi*a*(XA*dd + YA*(I-dd))
    force = np.dot(K, u_inf)
    Omega_0 = 8*pi*(a**3)*(XC*dd + YC*(I-dd))
    torque = np.dot(Omega_0, omega_inf) - 8*pi*a**3*YH*np.einsum('ijl,l,k,jk',
                    epsilon, d, d, E_inf)
    return force, torque


def get_force_torque_ellipsoid(a, b, c, dcm, u_inf, omega_inf, E_inf):
    if c == 'zero' and b == 'zero':
        print('RF and RJ ill-defined for b = c = 0.')
        return float('nan'), float('nan')
    asq = a**2
    bsq = b**2
    if c == 'zero':
        csq = 0.0
    else:
        csq = c**2

    asq_alpha_1 = (2*asq/3.0)*mpmath.elliprj(asq, bsq, csq, asq)
    bsq_alpha_2 = (2*bsq/3.0)*mpmath.elliprj(asq, bsq, csq, bsq)
    if c == 'zero':
        csq_alpha_3 = 0.0
    else:
        csq_alpha_3 = (2*csq/3.0)*mpmath.elliprj(asq, bsq, csq, csq)
    
    chi = 2*mpmath.elliprf(asq, bsq, csq)

    A = np.zeros((3,3))
    C = np.zeros((3,3))
    H_tilde = np.zeros((3,3,3))

    A[0,0] = 16*pi/(chi + asq_alpha_1)
    A[1,1] = 16*pi/(chi + bsq_alpha_2)
    A[2,2] = 16*pi/(chi + csq_alpha_3)
    A = shift_tensor2_dcm(A, dcm, forward=False)

    C[0,0] = (16*pi/3)*(bsq+csq)/(bsq_alpha_2 + csq_alpha_3)
    C[1,1] = (16*pi/3)*(csq+asq)/(asq_alpha_1 + csq_alpha_3)
    C[2,2] = (16*pi/3)*(asq+bsq)/(asq_alpha_1 + bsq_alpha_2)
    C = shift_tensor2_dcm(C, dcm, forward=False)

    H_tilde[0,1,2] =  (16*pi/3)*bsq/(bsq_alpha_2 + csq_alpha_3)
    H_tilde[0,2,1] = -(16*pi/3)*csq/(bsq_alpha_2 + csq_alpha_3)
    H_tilde[1,2,0] =  (16*pi/3)*csq/(asq_alpha_1 + csq_alpha_3)
    H_tilde[1,0,2] = -(16*pi/3)*asq/(asq_alpha_1 + csq_alpha_3)
    H_tilde[2,0,1] =  (16*pi/3)*asq/(asq_alpha_1 + bsq_alpha_2)
    H_tilde[2,1,0] = -(16*pi/3)*bsq/(asq_alpha_1 + bsq_alpha_2)
    H_tilde = shift_tensor3_dcm(H_tilde, dcm, forward=False)

    force = np.dot(A, u_inf)
    torque = np.dot(C, omega_inf) + np.einsum('ijk,jk', H_tilde, E_inf)
    return force, torque


axis, angle = get_rand_axis_angle()
dcm = axis_angle_to_dcm(axis, angle)
#dcm = np.identity(3)

#Streaming Velocity
u_inf = np.random.random((3,))
#u_inf = np.array([1, 0, 0]) #np.random.random((3,))
#omega_inf = np.zeros((3,))
#E_inf = np.zeros((3,3))
omega_inf = np.random.random((3,))
E_inf = np.random.random((3,3))
E_inf = (E_inf + E_inf.T)/2

print('Velocity field')
print('--------------')
print('Streaming velocity: ', u_inf)
print()
print('Angular velocity:   ', omega_inf)
print()
print('Strain rate tensor: ', E_inf[0,:])
print('                    ', E_inf[1,:])
print('                    ', E_inf[2,:])
print()

print('Direction cosine matrix: ', dcm[0,:])
print('                         ', dcm[1,:])
print('                         ', dcm[2,:])
print()

#Ellipsoid/sphere
print('Ellipsoid --> Sphere')
print('=============================\n')
a = 0.5
b = 0.5
c = 0.5

force, torque = get_force_torque_ellipsoid(a, b, c, dcm, u_inf,
        omega_inf, E_inf)
print('Ellipsoid ({0}, {1}, {2})'.format(a,b,c))
print('-------------------------')
print('Force:  ', force)
print('Torque: ', torque)
print()

force = 6*pi*a*u_inf
torque = 8*pi*a**3*omega_inf
print('Sphere ({0})'.format(a))
print('-------------------------')
print('Force (6*pi*r*u_inf):  ', force)
print('Torque (8*pi*r^3*omega_inf): ', torque)
print()

#Ellipsoid/oblate
print('Ellipsoid --> Oblate')
print('=============================\n')
a = 0.5
b = 0.5
c = 0.15

force, torque = get_force_torque_ellipsoid(a, b, c, dcm, u_inf,
        omega_inf, E_inf)
print('Ellipsoid ({0}, {1}, {2})'.format(a,b,c))
print('-------------------------')
print('Force:  ', force)
print('Torque: ', torque)
print()

d = rotate_vector_dcm(np.array([0,0,1]), dcm)
force, torque = get_force_torque_oblate(a, c, d, u_inf,
                        omega_inf, E_inf)
print('Oblate ({0}, {1}, {2})'.format(a,b,c))
print('-------------------------')
print('Force:  ', force)
print('Torque: ', torque)
print()

#Ellipsoid/prolate
print('Ellipsoid --> Prolate')
print('=============================\n')
a = 1.0
b = 0.5
c = 0.5

force, torque = get_force_torque_ellipsoid(a, b, c, dcm, u_inf,
        omega_inf, E_inf)
print('Ellipsoid ({0}, {1}, {2})'.format(a,b,c))
print('-------------------------')
print('Force:  ', force)
print('Torque: ', torque)
print()

d = rotate_vector_dcm(np.array([1,0,0]), dcm)
force, torque = get_force_torque_prolate(a, c, d, u_inf,
                           omega_inf, E_inf)
print('Prolate ({0}, {1}, {2})'.format(a,b,c))
print('-------------------------')
print('Force:  ', force)
print('Torque: ', torque)
print()


#Elliptic disc/circular disc
print('Elliptic disc --> Circular disc')
print('=============================\n')
a = 0.5
b = 0.5
c = 'zero'

force, torque = get_force_torque_ellipsoid(a, b, c, dcm, u_inf,
        omega_inf, E_inf)
print('Elliptic disc ({0}, {1}, {2})'.format(a,b,c))
print('-------------------------')
print('Force:  ', force)
print('Torque: ', torque)
print()

d = rotate_vector_dcm(np.array([0,0,1]), dcm)
force, torque = get_force_torque_oblate(a, c, d, u_inf,
                        omega_inf, E_inf)
print('Circular disc ({0}, {1}, {2})'.format(a,b,c))
print('-------------------------')
print('Force:  ', force)
print('Torque: ', torque)
print()


#Ellipsoid/needle
print('Ellipsoid --> Needle')
print('=============================\n')
a = 1.0
b = 'zero'
c = 'zero'

force, torque = get_force_torque_ellipsoid(a, b, c, dcm, u_inf,
        omega_inf, E_inf)
print('Ellipsoid ({0}, {1}, {2})'.format(a,b,c))
print('-------------------------')
print('Force:  ', force)
print('Torque: ', torque)
print()

d = rotate_vector_dcm(np.array([1,0,0]), dcm)
force, torque = get_force_torque_prolate(a, c, d, u_inf,
                           omega_inf, E_inf)
print('Needle ({0}, {1}, {2})'.format(a,b,c))
print('-------------------------')
print('Force:  ', force)
print('Torque: ', torque)
print()


