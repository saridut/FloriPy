#!/usr/bin/env python

import math
import csv
import numpy as np
from floripy.mathutils import xform as tr
from floripy.mathutils.linalg import unitized
from .miura_sheet_trajectory import MiuraSheetTrajectory


def get_phi_theta(v):
    '''
    v: (3,) ndarray
    Returns phi and theta in degrees.
    phi: Angle measured from the y-axis of the projection on the xy-plane
            -pi<=phi<=pi
    theta: Angle measured from the z-axis, -pi/2<=theta<=pi/2
    '''
    vec = unitized(v)
    phi = math.atan2(vec[0], vec[1])
    theta = math.acos(vec[2])
    phi_deg = math.degrees(phi)
    theta_deg = math.degrees(theta)
    return phi_deg, theta_deg


def basic_data(fn_traj, fn_model, fn_data):
    #Open trajectory file and calculate
    mt = MiuraSheetTrajectory(fn_traj, fn_model)
    num_frames = len(mt)
    print('Number of frames:  ', num_frames)
    field_names = ['time', 'beta',
                    'chord', 'chord_normalized',
                    'span', 'span_normalized',
                    'aspect_ratio', 'aspect_ratio_normalized',
                    'theta_director', 'theta_codirector', 'theta_bidirector',
                    'phi_director', 'phi_codirector','phi_bidirector',
                    'roll', 'yaw', 'pitch',
                    'comx', 'comy', 'comz',
                    'directorx', 'directory', 'directorz',
                    'codirectorx', 'codirectory', 'codirectorz',
                    'bidirectorx', 'bidirectory', 'bidirectorz']
    data = {}
    with open(fn_data, 'w') as fh_data:
        writer = csv.DictWriter(fh_data, field_names)
        writer.writeheader()
        for k in range(num_frames):
            print('Frame: ', k) if k%100==0 else None
            time, ms = mt.get_frame(k)
            data['time'] = time
            data['beta'] = math.degrees(ms.beta)
            data['chord'] = ms.chord
            data['chord_normalized'] = ms.chord/ms.max_chord
            data['span'] = ms.span
            data['span_normalized'] = ms.span/ms.max_span
            data['aspect_ratio'] = ms.aspect_ratio
            data['aspect_ratio_normalized'] = ms.aspect_ratio/ms.max_aspect_ratio
    
            director = ms.director
            codirector = ms.codirector
            bidirector = ms.bidirector
    
            data['directorx'] = director[0]
            data['directory'] = director[1]
            data['directorz'] = director[2]
    
            data['codirectorx'] = codirector[0]
            data['codirectory'] = codirector[1]
            data['codirectorz'] = codirector[2]
    
            data['bidirectorx'] = bidirector[0]
            data['bidirectory'] = bidirector[1]
            data['bidirectorz'] = bidirector[2]
    
            #Theta: Angle measured from the z-axis, -pi/2<=theta<=pi/2
            #Phi: Angle measured from the y-axis of the projection on the xy-plane
            #-pi<=phi<=pi
            phi_director, theta_director = get_phi_theta(director)
            phi_codirector, theta_codirector = get_phi_theta(codirector)
            phi_bidirector, theta_bidirector = get_phi_theta(bidirector)
    
            data['theta_director'] = theta_director
            data['theta_codirector'] = theta_codirector
            data['theta_bidirector'] = theta_bidirector
    
            data['phi_director'] = phi_director
            data['phi_codirector'] = phi_codirector 
            data['phi_bidirector'] = phi_bidirector
    
            ori = ms.orientation
            ori_euler_body = np.rad2deg(tr.quat_to_euler(ori, seq='XYZ', world=False))
            data['roll'], data['yaw'], data['pitch'] = tuple(ori_euler_body)
            data['comx'], data['comy'], data['comz'] = tuple(ms.com)
    
            writer.writerow(data)
    mt.close()
