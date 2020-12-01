#!/usr/bin/env python
import math
import numpy as np
from xform import *


def test_axis_angle_conversion():
    print('=====================')
    print('axis-angle conversion')
    print('=====================')
    axes = []; angles = []
    axis, angle = get_rand_axis_angle()
    axes.append(axis)
    angles.append(angle)
    axes.extend([np.array([0.0, 0.0, 1.0])]*3)
    angles.extend([0.0, math.pi/2, math.pi])
    for axis, angle in zip(axes, angles):
        print('axis: ', axis)
        print('angle: ', math.degrees(angle))
        print() 

        print('axis-angle --> dcm')
        dcm = axis_angle_to_dcm(axis, angle)
        u, theta = dcm_to_axis_angle(dcm)
        if (np.allclose(u, axis) and 
                math.isclose(theta, angle, abs_tol=1e-12, rel_tol=1e-12)):
            print('OK')
        else:
            print('dcm: ')
            print(dcm[0,:]); print(dcm[1,:]); print(dcm[2,:])
            print('axis: ', u)
            print('angle: ', math.degrees(theta))
        print() 

        print('axis-angle --> quat')
        q = axis_angle_to_quat(axis, angle)
        u, theta = quat_to_axis_angle(q)
        if (np.allclose(u, axis) and 
                math.isclose(theta, angle, abs_tol=1e-12, rel_tol=1e-12)):
            print('OK')
        else:
            print('quat: ', q)
            print('axis: ', u)
            print('angle: ', math.degrees(theta))
        print() 

        print('axis-angle --> euler')
        euler = axis_angle_to_euler(axis, angle)
        u, theta = euler_to_axis_angle(euler)
        if (np.allclose(u, axis) and 
                math.isclose(theta, angle, abs_tol=1e-12, rel_tol=1e-12)):
            print('OK')
        else:
            print('euler: ', np.rad2deg(euler))
            print('axis: ', u)
            print('angle: ', math.degrees(theta))
        print('++++++++++++++++++++++++++++++++++++++++++++++++++++')
        print() 


def test_quat_conversion():
    print('=====================')
    print('Quaternion conversion')
    print('=====================')
    axes = []; angles = []
    axis, angle = get_rand_axis_angle()
    axes.append(axis)
    angles.append(angle)
    axes.extend([np.array([0.0, 0.0, 1.0])]*3)
    angles.extend([0.0, math.pi/2, math.pi])
    for axis, angle in zip(axes, angles):
        q = axis_angle_to_quat(axis, angle)
        print('quat: ', q)
        print() 

        print('quat --> dcm')
        dcm = quat_to_dcm(q)
        p = dcm_to_quat(dcm)
        if not np.allclose(q, p):
            print('dcm: ')
            print(dcm[0,:]); print(dcm[1,:]); print(dcm[2,:])
            print('quat: ', p)
        else:
            print('OK')
        print() 

        print('quat --> axis-angle')
        u, theta = quat_to_axis_angle(q)
        p = axis_angle_to_quat(u, theta)
        if not np.allclose(q, p):
            print('axis: ', u)
            print('angle: ', math.degrees(theta))
            print('quat: ', p)
        else:
            print('OK')
        print() 

        print('quat --> euler')
        euler = quat_to_euler(q)
        p = euler_to_quat(euler)
        if not np.allclose(q, p):
            print('euler: ', np.rad2deg(euler))
            print('quat: ', p)
        else:
            print('OK')
        print('++++++++++++++++++++++++++++++++++++++++++++++++++++')
        print() 


def test_euler_conversion():
    print('=====================')
    print('Euler conversion')
    print('=====================')
    phi = 2.0*np.pi*np.random.random() - np.pi
    theta = np.pi*np.random.random() - np.pi/2
    psi = 2.0*np.pi*np.random.random() - np.pi
    ea = [np.array([phi, theta, psi]),
            np.array([0, 0, 0]),
            np.array([0, math.pi/2, 0]),
            np.array([math.pi/2, 0, 0]),
            np.array([0, 0, math.pi/2])]

    for euler in ea:
        print('euler: ', np.rad2deg(euler))
        print() 

        print('euler --> dcm')
        dcm = euler_to_dcm(euler)
        e = dcm_to_euler(dcm)
        if not np.allclose(euler, e):
            print('dcm: ')
            print(dcm[0,:]); print(dcm[1,:]); print(dcm[2,:])
            print('euler: ', np.rad2deg(e))
        else:
            print('OK')
        print() 

        print('euler --> axis-angle')
        u, theta = euler_to_axis_angle(euler)
        e = axis_angle_to_euler(u, theta)
        if not np.allclose(euler, e):
            print('axis: ', u)
            print('angle: ', math.degrees(theta))
            print('euler: ', np.rad2deg(e))
        else:
            print('OK')
        print() 

        print('euler --> quat')
        q = euler_to_quat(euler)
        e = quat_to_euler(q)
        if not np.allclose(euler, e):
            print('quat: ', q)
            print('euler: ', np.rad2deg(e))
        else:
            print('OK')
        print('++++++++++++++++++++++++++++++++++++++++++++++++++++')
        print() 


def test_dcm_conversion():
    print('=====================')
    print('Dcm conversion')
    print('=====================')

    axis, angle = get_rand_axis_angle()
    axes = np.identity(3)
    axes = rotate_vector_axis_angle(np.identity(3), axis, angle)
    dcm = dcm_from_axes(np.identity(3), axes)
    print(mat_is_dcm(dcm))

    print('dcm: ')
    print(dcm[0,:]); print(dcm[1,:]); print(dcm[2,:])
    print() 

    print('dcm --> euler')
    euler = dcm_to_euler(dcm)
    D = euler_to_dcm(euler)
    if not np.allclose(D, dcm):
        print('euler: ', np.rad2deg(euler))
        print('dcm: ')
        print(D[0,:]); print(D[1,:]); print(D[2,:])
    else:
        print('OK')
    print() 

    print('dcm --> axis-angle')
    u, theta = dcm_to_axis_angle(dcm)
    D = axis_angle_to_dcm(u, theta)
    if not np.allclose(D, dcm):
        print('axis: ', u)
        print('angle: ', math.degrees(theta))
        print('dcm: ')
        print(D[0,:]); print(D[1,:]); print(D[2,:])
    else:
        print('OK')
    print() 

    print('dcm --> quat')
    q = dcm_to_quat(dcm)
    D = quat_to_dcm(q)
    if not np.allclose(D, dcm):
        print('quat: ', q)
        print('dcm: ')
        print(D[0,:]); print(D[1,:]); print(D[2,:])
    else:
        print('OK')
    print('++++++++++++++++++++++++++++++++++++++++++++++++++++')
    print() 


def test_rot_shift_axis_angle():
    axis, angle = get_rand_axis_angle()
    v = np.random.random((3,3))
    v_rot = rotate_vector_axis_angle(v, axis, angle)
    v_sft = shift_vector_axis_angle(v_rot, axis, angle, forward=True)
    if np.allclose(v, v_sft):
        print('OK')
    else:
        print(v)
        print(v_sft)


def test_rot_shift_quat():
    q = get_rand_quat()
    v = np.random.random((3,3))
    v_rot = rotate_vector_quat(v, q)
    v_sft = shift_vector_quat(v_rot, q, forward=True)
    if np.allclose(v, v_sft):
        print('OK')
    else:
        print(v)
        print(v_sft)


def test_rot_shift_dcm():
    pass


def test_rot_shift_euler():
    q = get_rand_quat()
    euler = quat_to_euler(q, seq='XYZ', world=True)
    v = np.random.random((3,3))
    v_rot = rotate_vector_euler(v, euler, seq='XYZ', world=True)
    v_sft = shift_vector_euler(v_rot, euler, seq='XYZ', world=True, forward=True)
    if np.allclose(v, v_sft):
        print('OK')
    else:
        print(v)
        print(v_sft)


def test_intra_euler():
    seq = ['XYZ', 'XZY', 'YXZ', 'YZX', 'ZXY', 'ZYX']
    for each in seq:
        euler = 2*math.pi*(2*np.random.random((3,)) - 1)
        euler_in = euler_to_euler(euler, each, True, each, True)
        euler_out = euler_to_euler(euler_in, each, True, each, True)
        if not np.allclose(euler_in, euler_out):
            print(each, 'FAILED')



def test_align():
    pass


if __name__ == '__main__':
    test_axis_angle_conversion()
    test_quat_conversion()
    test_euler_conversion()
    test_dcm_conversion()
    test_rot_shift_axis_angle()
    test_rot_shift_quat()
    test_rot_shift_euler()
    test_intra_euler()
