#!/usr/bin/env python
import math
import numpy as np
import scipy.linalg as sla

from floripy.mathutils import xform as tr
from floripy.mathutils import linalg as mla


def get_sector_angles(vertices_xy, center=[0.0,0.0]):
    vertex_degree = len(vertices_xy)
    edge_unit_vectors = []
    for vertex in vertices_xy:
        edge_vector_x = vertex[0]-center[0]
        edge_vector_y = vertex[1]-center[1]
        edge_vector_mag = math.sqrt(edge_vector_x**2 + edge_vector_y**2)
        edge_unit_vectors.append(
                [edge_vector_x/edge_vector_mag,
                edge_vector_y/edge_vector_mag]
                )
    for k in range(vertex_degree-1):
        euv_k = edge_unit_vectors[k]
        euv_kp1 = edge_unit_vectors[k+1]
        sa = math.acos(euv_k[0]*euv_kp1[0] + euv_k[1]*euv_kp1[1])
        sector_angles.append(sa)

    #Calculate the last sector angle
    sa = 2*math.pi - sum(sector_angles)
    sector_angles.append(sa)
    return sector_angles


def get_degree4_dihedral_angles(sa, beta):
    #Check that all four sector angles are positive
    for i in range(4):
        if sa[i] <= 0:
            raise ValueError(
                    'Sector angles must be positive.\n' +
                    'Found sa[{0}] = {1}'.format(i, sa[i])
                    )

    da = np.zeros((4,))
    da[0] = beta
    #Avoid division by zero
    if math.isclose(sa[0], sa[1], rel_tol=1e-14):
        raise ValueError(
                'Sector angles sa[0] = sa[1] = {0} (mach. prec.).'.format(sa[0])
                + 'These angles have to be different.')
    #Avoid multiple values for dihedral angles
    elif math.isclose(abs(da[0]), math.pi, rel_tol=1e-14):
        raise ValueError('Dihedral angle da[0] = {0} (mach. prec.).'.format(da[0])
                + 'This has to be different from +/-pi.')
    else:
        da[1] = 2*math.atan(math.tan(da[0]/2)
                        * math.sin((sa[0]+sa[1])/2)
                        / math.sin((sa[0]-sa[1])/2)
                        )
    da[2] = -da[0]
    da[3] = da[1]
    return da


def get_dihedral_angle_between_polys(poly1, poly2, edge):
    '''
    Returns the dihedral angle between two polygons sharing an edge.
    '''
    edge_vector = edge[1,:] - edge[0,:]
    edge_unit_vector = edge_vector/np.linalg.norm(edge_vector)

    poly1_edge1 = poly1[1,:] - poly1[0,:]
    poly1_edge2 = poly1[2,:] - poly1[1,:]

    poly2_edge1 = poly2[1,:] - poly2[0,:]
    poly2_edge2 = poly2[2,:] - poly2[1,:]

    poly1_normal = np.cross(poly1_edge1, poly1_edge2)
    poly1_unit_normal = poly1_normal/np.linalg.norm(poly1_normal)
    poly2_normal = np.cross(poly2_edge1, poly2_edge2)
    poly2_unit_normal = poly2_normal/np.linalg.norm(poly2_normal)
    ydir = np.cross(edge_unit_vector, poly1_unit_normal)
    yhat = ydir/np.linalg.norm(ydir)
    d = np.dot(poly1_unit_normal, poly2_unit_normal)
    q = np.dot(poly2_unit_normal, yhat)
    angle = math.atan2(q, d)
    return angle


def polygon(vertices):
    num_vertices = vertices.shape[0]
    body_axes = np.zeros((3,3))
    dir_x = vertices[1,:] - vertices[0,:]
    body_axes[0,:] = dir_x/np.linalg.norm(dir_x)
    edge_0 = vertices[1,:] - vertices[0,:]
    edge_1 = vertices[-1,:] - vertices[0,:]
    dir_y = np.cross(edge_0, edge_1)
    body_axes[1,:] = dir_y/np.linalg.norm(dir_y)
    dir_z = np.cross(body_axes[0,:], body_axes[1,:])
    body_axes[2,:] = dir_z/np.linalg.norm(dir_z)
    
    world_axes = np.identity(3)
    dcm = np.zeros((3,3))
    for m in range(3):
        for n in range(3):
            dcm[m,n] = np.dot(body_axes[m,:], world_axes[n,:])
    com = vertices.mean(axis=0)
    body_vertices = np.zeros((num_vertices,3))
    for i in range(num_vertices):
        vertex = vertices[i,:] - com
        body_vertices[i,:] = np.dot(dcm, vertex)
    return com, dcm, body_vertices


def fit_ellipse(vertices):
    '''
    Fits an ellipse to a parallelogram.

    '''
    com = np.mean(vertices, axis=0)
    for k in range(vertices.shape[0]):
        vertices[k,:] -= com

    edge_0 = vertices[1,:] - vertices[0,:]
    edge_1 = vertices[2,:] - vertices[1,:]
    ell_0 = np.linalg.norm(edge_0)
    ell_1 = np.linalg.norm(edge_1)
#   print('ell_0: ', ell_0, 'ell_1: ', ell_1)
    alpha = math.acos(np.dot(edge_0, edge_1)/(ell_0*ell_1))
#   print('alpha: ', math.degrees(alpha))
    phi = math.pi/2 - alpha
#   print('phi: ', math.degrees(phi))
    #Basis vectors of the parallelogram frame (pca) in terms of the world frame
    xbv_pca = edge_0/ell_0
    ybv_pca = mla.unitized(np.cross(edge_0, edge_1))
    zbv_pca = mla.unitized(np.cross(xbv_pca, ybv_pca))
    dcm_pca = tr.dcm_from_axes(np.identity(3),
                    np.array([xbv_pca, ybv_pca, zbv_pca]))

    h = ell_1*math.sin(alpha)
#   print('h: ', h)
    radius = ell_0/2
#   print('radius: ', radius)
    scale_xform = np.array([[1, 0], [0, h/ell_0]])
#   scale_xform = np.identity(2)
    shear_xform = np.array([[1, math.tan(phi)], [0, 1]])
    F = np.dot(shear_xform, scale_xform)
#   print('F: ', F)
    #Polar decomposition (2D)
    rotmat, extmat = sla.polar(F, side='right')
#   print(rotmat)
#   print(extmat)
    eigval, eigvec = sla.eigh(extmat)
    #Rotate the eigen vectors using rotmat
    eigvec_rot = np.dot(rotmat, eigvec)
    #Compare eigenvalue magnitudes to assign semi-major axis length a and
    #semi-minor axis length c. Make 3D vectors out of the 2D rotated eigen
    #vectors. These are the ellipse frame (eca) basis vectors in pca.
    if abs(eigval[0]) > abs(eigval[1]):
        a = radius*eigval[0]
        c = radius*eigval[1]
        xbv_eca_pca = np.array([eigvec_rot[0,0], 0, eigvec_rot[1,0]])
        zbv_eca_pca = np.array([eigvec_rot[0,1], 0, eigvec_rot[1,1]])
    elif abs(eigval[0]) < abs(eigval[1]):
        a = radius*eigval[0]
        c = radius*eigval[1]
        xbv_eca_pca = np.array([eigvec_rot[0,1], 0, eigvec_rot[1,1]])
        zbv_eca_pca = np.array([eigvec_rot[0,0], 0, eigvec_rot[1,0]])
    else:
        #Both eigen values equal
        a = radius*eigval[0]
        c = radius*eigval[0]
        xbv_eca_pca = np.array([eigvec_rot[0,0], 0, eigvec_rot[1,0]])
        zbv_eca_pca = np.array([eigvec_rot[0,1], 0, eigvec_rot[1,1]])
        #Check using the scalar triple product. ybv_pca is the same as
        #ybv_eca_pca
        scalar_trip_prod = np.dot(np.cross(xbv_eca_pca, ybv_pca), zbv_eca_pca)
        if scalar_trip_prod < 0.0:
            zbv_eca_pca, xbv_eca_pca = xbv_eca_pca, zbv_eca_pca

    #Shift the rotated eigenvectors from pca to world.
    xbv_eca = tr.shift_vector_dcm(xbv_eca_pca, dcm_pca, forward=False)
    zbv_eca = tr.shift_vector_dcm(zbv_eca_pca, dcm_pca, forward=False)
    #ybv remains the same
    ybv_eca = ybv_pca
    wca = np.identity(3)
    dcm = tr.dcm_from_axes(wca, np.array([xbv_eca, ybv_eca, zbv_eca]))
    return float(a), float(c), dcm

