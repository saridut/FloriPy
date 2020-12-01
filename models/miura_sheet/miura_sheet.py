#!/usr/bin/env python

import math
import os.path
import numpy as np
from lxml import etree as et
import h5py

from floripy.file_formats import yamlio
from floripy.mathutils import xform as tr
from floripy.mathutils import geometry as geom
from floripy.mathutils.linalg import unitized


class MiuraSheet(object):
    def __init__(self, nverts_z, nverts_x, verts, faces, joint_edges):
        '''
        Parameters
        ----------
        nverts_z : int
            Number of vertices along the z-direction. The z-direction is the fastest 
            varying axis for describing the `verts` array (see below).
        nverts_x : int
            Number of vertices along x-direction. The x-direction is the slowest 
            varying axis for describing the `verts` array (see below). There is
            no variable `nverts_y`, it has an assumed constant value of 1.
        verts : (N,3) ndarray, dtype float
            Coordinates of the vertices in the world frame or the c.o.m. frame.
            Row i of `verts` contains the coordinates of vertex i.
        faces : (N,4) ndarray, dtype int
            Row i contains the list of vertices that form face i. The order of
            the vertices define the direction of the normal.
        joint_edges : dict


        Returns
        -------
        A MiuraSheet instance

        '''
        #Vertices
        self.nverts_z = nverts_z
        self.nverts_x = nverts_x
        self.nverts = self.nverts_z*self.nverts_x
        self.verts = verts

        #Faces
        self.nfaces_z = nverts_z - 1
        self.nfaces_x = nverts_x - 1
        self.nfaces = self.nfaces_z*self.nfaces_x
        self.faces = faces

        #Horizontal joint edges
        self.nhje_z = self.nfaces_z - 1
        self.nhje_x = self.nfaces_x
        self.nhje = self.nhje_z*self.nhje_x

        #Vertical joint edges
        self.nvje_z = self.nfaces_z
        self.nvje_x = self.nfaces_x - 1
        self.nvje = self.nvje_z*self.nvje_x

        #All joint edges
        self.njoint_edges = self.nhje + self.nvje
        self.joint_edges = joint_edges

        assert self.nverts == self.verts.shape[0]
        assert self.nfaces == self.faces.shape[0]
        assert self.njoint_edges == len(self.joint_edges)

        #c.o.m. of the sheet
        self.com = np.mean(self.verts, axis=0)
        for k in range(self.nverts):
            self.verts[k,:] -= self.com

        #Side a
        a_vec = self.verts[self.nverts_z,:] - self.verts[0,:]
        a_len = np.linalg.norm(a_vec)
        a_uv = a_vec/a_len

        #Side b
        b_vec = self.verts[1,:] - self.verts[0,:]
        b_len = np.linalg.norm(b_vec)
        b_uv = b_vec/b_len

        #Sector angle alpha
        self.alpha = math.acos(np.dot(a_uv,b_uv))

        #Span, chord, and aspect ratio
        span_vec = self.verts[2*self.nverts_z,:] - self.verts[0,:]
        self.span = np.linalg.norm(span_vec)
        chord_vec = self.verts[2,:] - self.verts[0,:]
        self.chord = np.linalg.norm(chord_vec)
        self.aspect_ratio = self.span/self.chord
        self.max_chord = 2*b_len
        self.max_span = 2*a_len*math.sin(self.alpha)
        self.max_aspect_ratio = self.max_span/self.max_chord

        #Dihedral angle beta
        poly1 = self.verts[tuple(self.faces[0,:]),:]
        poly2 = self.verts[tuple(self.faces[self.nfaces_z,:]),:]
        bv = self.joint_edges[self.nhje]['bv']
        ev = self.joint_edges[self.nhje]['ev']
        edge = self.verts[(bv, ev),:]
        self.beta = geom.get_dihedral_angle_between_polys(poly1, poly2, edge)

        #Director coordinate axes
        self.director = unitized(chord_vec)
        if math.isclose(self.span, 0.0, rel_tol=1e-12):
            mid_vec = (self.verts[0,:]+self.verts[2,:])/2
            self.codirector = unitized(self.verts[1,:]-mid_vec)
            self.bidirector = unitized(np.cross(self.director,
                                    self.codirector))
        else:
            self.bidirector = unitized(-span_vec)
            self.codirector = unitized(np.cross(self.bidirector,
                                    self.director))
        dca = np.zeros((3,3))
        dca[0,:] = np.copy(self.director)
        dca[1,:] = np.copy(self.codirector)
        dca[2,:] = np.copy(self.bidirector)
        dcm = tr.dcm_from_axes(np.identity(3), dca)
        self.orientation = tr.dcm_to_quat(dcm)

        #Additional derived measures
        self.bbone_theta = math.acos(self.chord/(2*b_len))
        self.bbone_delta = math.acos(1-0.5*(self.chord/b_len)**2)
        self.thickness = b_len*math.sin(self.bbone_theta)


    @classmethod
    def create(cls, a, b, alpha, beta, eq_angle=0.0, H=0.0, tiling=(1,1)):
        '''
        Creates a MiuraSheet instance with the c.o.m. at the origin.
        Tiling: (#vert, #hor), i.e. (depth, width)
        '''
        #Sector and dihedral angles
        assert math.isclose(alpha, math.pi/2, rel_tol=1e-14) == False
        sa = np.zeros((4,))
        sa[0] = alpha
        sa[1] = math.pi - alpha
        sa[2] = sa[1]
        sa[3] = sa[0]
        da = geom.get_degree4_dihedral_angles(sa, beta)
        da = da.tolist()

        #Parallelogram
        sin_alpha = math.sin(alpha)
        cos_alpha = math.cos(alpha)
        p = b*sin_alpha
        q = a*sin_alpha
        r = b*cos_alpha
        s = a*cos_alpha
        h = p*math.sin(-da[1]/2)
        l = p*math.cos(-da[1]/2)
        u = math.sqrt(l**2 + r**2)
        v = q*math.cos(da[0]/2)
        w = math.sqrt(a**2-v**2)

        #Tiling of the unit cell
        ntiles_z, ntiles_x = tiling
        nfaces_z = 2*ntiles_z
        nfaces_x = 2*ntiles_x
        nfaces = nfaces_z*nfaces_x
        nverts_z = nfaces_z + 1
        nverts_x = nfaces_x + 1
        nverts = nverts_z*nverts_x
        nhje_z = nfaces_z - 1
        nhje_x = nfaces_x
        nhje = nhje_z*nhje_x
        nvje_z = nfaces_z
        nvje_x = nfaces_x - 1
        nvje = nvje_z*nvje_x
        njoint_edges = nhje + nvje

        #Create the vertices
        verts = np.zeros((nverts, 3))
        for j in range(nverts_x):
            for i in range(nverts_z):
                k = j*nverts_z + i
                xloc = 'even' if j%2==0 else 'odd'
                zloc = 'even' if i%2==0 else 'odd'
                if zloc == 'even' and xloc == 'even':
                    verts[k,0] = j*v
                    verts[k,1] = 0.0
                    verts[k,2] = i*u
                elif zloc == 'odd' and xloc == 'odd':
                    verts[k,0] = j*v
                    verts[k,1] = h
                    verts[k,2] = w + i*u
                elif zloc == 'odd' and xloc == 'even':
                    verts[k,0] = j*v
                    verts[k,1] = h
                    verts[k,2] = i*u
                elif zloc == 'even' and xloc == 'odd':
                    verts[k,0] = j*v
                    verts[k,1] = 0.0
                    verts[k,2] = w + i*u

        #Create the faces
        faces = np.zeros((nfaces, 4), dtype=np.int32)
        for j in range(nfaces_x):
            for i in range(nfaces_z):
                k = j*nfaces_z + i
                kv_ul = j*nverts_z + i
                kv_ll = kv_ul + 1
                kv_lr = kv_ll + nverts_z
                kv_ur = kv_lr - 1
                faces[k,:] = np.array([kv_ul, kv_ll, kv_lr, kv_ur],
                                        dtype=np.int32)

        #Create the joint_edges
        joint_edges = {}
        #Joints across the horizontal edges
        for j in range(nhje_x):
            for i in range(nhje_z):
                k = j*nhje_z + i
                if i%2 == 0:
                    angle = da[1] if j%2 == 0 else da[3]
                else:
                    angle = -da[1] if j%2 == 0 else -da[3]
                joint_edges[k] = {'bv': ((j+1)*nverts_z + i + 1),
                                'ev': (j*nverts_z + i + 1),
                                'pf': (j*nfaces_z + i),
                                'sf': (j*nfaces_z + i + 1),
                                'angle': angle,
                                'H': 0.0,
                                'eq_angle': 0.0
                                }
        #Joints across the vertical edges
        for j in range(nvje_x):
            for i in range(nvje_z):
                k = nhje + j*nvje_z + i
                if i%2 == 0:
                    angle = da[0] if j%2 == 0 else -da[0]
                else:
                    angle = da[2] if j%2 == 0 else -da[2]
                joint_edges[k] = {'bv': ((j+1)*nverts_z + i),
                                  'ev': ((j+1)*nverts_z + i + 1),
                                  'pf': (j*nfaces_z + i),
                                  'sf': ((j+1)*nfaces_z + i),
                                  'angle': angle,
                                  'H': 0.0,
                                  'eq_angle': 0.0
                                  }

        #Bring com to origin
        com_verts = verts.mean(axis=0)
        for k in range(nverts):
            verts[k,:] -= com_verts
        ms = cls(nverts_z, nverts_x, verts, faces, joint_edges)
        return ms


    @classmethod
    def from_df_mbs(cls, df_mbs, modelspec):
        '''
        Creates a MiuraSheet instance from df_mbs and modelspec. The c.o.m.
        position and orientation is as calculated from df_mbs.

        Parameters
        ----------
        df_mbs : dict
            keywords: 'body_coms', 'body_orientations'
        modelspec : dict

        Returns
        -------
        ms : An instance of MiuraSheet

        '''
        #Extract from df_mbs
        body_coms = df_mbs['body_coms']
        body_oris = df_mbs['body_orientations']

        #Extract from modelspec
        bodies = modelspec['bodies']
        joints = modelspec['joints']
        internal = modelspec['{internal}']
        faces = np.asarray(internal['faces'], dtype='i4')
        joint_edges = internal['joint_edges']
        nverts_z = internal['nverts_z']
        nverts_x = internal['nverts_x']
        vert_map = np.asarray(internal['vert_map'], dtype='i4')
        nverts = nverts_z*nverts_x
        verts = np.zeros((nverts,3))

        for k in range(nverts):
            iface = vert_map[k,0]
            ivert = vert_map[k,1]
            ibody = iface + 1
            body_vert = np.asarray(bodies[ibody]['parallelogram']['vertices'][ivert])
            body_com = body_coms[iface]
            body_ori = body_oris[iface]
            vert = tr.shift_vector_quat(body_vert, body_ori, forward=False)
            verts[k,:] = vert + body_com

        ms = cls(nverts_z, nverts_x, verts, faces, joint_edges)
        return ms


    @classmethod
    def from_df_traj(cls, df_traj, modelspec):
        '''
        Parameters
        ----------
        df_traj : dict
            keywords: 'vertices', 'com'
        modelspec : dict

        Returns
        -------
        ms : An instance of MiuraSheet

        '''
        #Extract from df_traj. verts are the vertices w.r.t to the c.o.m frame
        #of reference.
        verts = df_traj['vertices']
        com = df_traj['com']

        #Extract from modelspec
        internal = modelspec['{internal}']
        faces = np.asarray(internal['faces'], dtype='i4')
        joint_edges = internal['joint_edges']
        nverts_z = internal['nverts_z']
        nverts_x = internal['nverts_x']
        nverts = nverts_z*nverts_x

        #Create a MiuraSheet instance and set its c.o.m
        ms = cls(nverts_z, nverts_x, verts, faces, joint_edges)
        ms.set_com(com)
        return ms


    def to_mbs(self, fn_yaml):
        face_prop = []
        #Create the bodies
        bodies = {}
        for k in range(self.nfaces):
            face_verts = self.verts[tuple(self.faces[k,:]),:]
            com, dcm, body_verts = geom.polygon(face_verts)
            face_prop.append((com, dcm, body_verts))
            bodies[k+1] = {'parallelogram': {'vertices': body_verts.tolist()}}

        #Create the joints
        joints = {}
        jid = 1
        #Root joint for Body 1
        dcm = face_prop[0][1]
        body_verts = face_prop[0][2]
        #Get c.o.m of Body 1 in the world frame
        body1_com = face_prop[0][0] + self.com
        #shifting -c.o.m. of body 1 from world frame to body_1 frame
        to_vert = tr.shift_vector_dcm(-body1_com, dcm, forward=True)
        #convert the dcm to quaternion.
        q = tr.dcm_to_quat(dcm)
        root_joint = {'joint_type': 'root',
                      'to': to_vert.tolist(),
                      'orientation':
                            {'repr': 'quat',
                              'quat': q.tolist()
                            }
                     }
        joints[jid] = root_joint

        #Revolute joints
        for k in range(self.njoint_edges):
            jid += 1
            bv = self.joint_edges[k]['bv']
            ev = self.joint_edges[k]['ev']

            pf = self.joint_edges[k]['pf']
            pf_com = face_prop[pf][0]
            pf_dcm = face_prop[pf][1]
            pf_edge = np.zeros((2,3))
            pf_edge[0,:] = tr.shift_vector_dcm((self.verts[bv,:] - pf_com),
                            pf_dcm, forward=True)
            pf_edge[1,:] = tr.shift_vector_dcm((self.verts[ev,:] - pf_com),
                            pf_dcm, forward=True)

            sf = self.joint_edges[k]['sf']
            sf_com = face_prop[sf][0]
            sf_dcm = face_prop[sf][1]
            sf_edge = np.zeros((2,3))
            sf_edge[0,:] = tr.shift_vector_dcm((self.verts[bv,:] - sf_com),
                            sf_dcm, forward=True)
            sf_edge[1,:] = tr.shift_vector_dcm((self.verts[ev,:] - sf_com),
                            sf_dcm, forward=True)

            angle = geom.get_dihedral_angle_between_polys(
                        self.verts[tuple(self.faces[pf,:]),:],
                        self.verts[tuple(self.faces[sf,:]),:],
                        np.array([self.verts[bv,:], self.verts[ev,:]])) 
            eq_angle = self.joint_edges[k]['eq_angle']

            joint = {'joint_type': 'revolute',
                     'pf': pf + 1,
                     'sf': sf + 1,
                     'pf_edge': pf_edge.tolist(),
                     'sf_edge': sf_edge.tolist(),
                     'angle': angle,
                     'H': self.joint_edges[k]['H'],
                     'eq_angle': eq_angle
                     }
            joints[jid] = joint


        #Data for reconstructing the model from MBS data
        internal = {}
        internal['nverts_z'] = self.nverts_z
        internal['nverts_x'] = self.nverts_x
        internal['vert_map'] = []
        for k in range(self.nverts):
            q, r = divmod(k, self.nverts_z)
            if (q <= (self.nfaces_x-1)) and (r <= (self.nfaces_z-1)):
                ivert = 0
            elif (q <= (self.nfaces_x-1)) and (r > (self.nfaces_z-1)):
                r = self.nfaces_z - 1
                ivert = 1
            elif (q > (self.nfaces_x-1)) and (r <= (self.nfaces_z-1)):
                q = self.nfaces_x - 1
                ivert = 3
            else:
                q = self.nfaces_x - 1
                r = self.nfaces_z - 1
                ivert = 2
            iface = q*self.nfaces_z + r
            internal['vert_map'].append([iface, ivert])

        internal['faces'] = self.faces.tolist()
        internal['joint_edges'] = self.joint_edges

        #Write to yaml file
        yamlio.write({'bodies': bodies, 'joints': joints,
                        '{internal}': internal}, fn_yaml)


    def set_com(self, loc):
        self.com = loc


    def align(self, axes):
        '''
        Changes orienation.
        axes : (3,3) ndarray, dtype float
            Orthonormal set of unit vectors. The rows represent the x, y, and
            z-basis vectors respectively in the world frame.
        '''
        dcm = tr.dcm_from_axes(np.identity(3), axes)
        q = tr.dcm_to_quat(dcm)
        self.set_orientation(q)


    def set_orientation(self, ori):
        '''
        ori: (4,) ndarray representing a unit quaternion
        '''
        q_ini = np.copy(self.orientation)
        q_fin = ori
        #Vector rotation
        q = tr.get_quat_prod(q_fin, tr.invert_quat(q_ini))
        self.verts = tr.rotate_vector_quat(self.verts, q)
        #Update the director, codirector, and bidirector
        #Transform the axes to the world frame using the orientation
        self.director = tr.shift_vector_quat(np.array([1,0,0]), q_fin,
                forward=False)
        self.codirector = tr.shift_vector_quat(np.array([0,1,0]), q_fin,
                forward=False)
        self.bidirector = tr.shift_vector_quat(np.array([0,0,1]), q_fin,
                forward=False)
        #Update the orientation
        self.orientation = ori


    def write_mesh(self, fn_xmf):
        xdmf = et.XML('''<?xml version="1.0"?>
                <!DOCTYPE Xdmf SYSTEM "Xdmf.dtd" []>
                <Xdmf Version="3.0" xmlns:xi="http://www.w3.org/2001/XInclude"/>
                ''')
        doc = et.ElementTree(element=xdmf)
        domain = et.SubElement(xdmf, 'Domain')
        #A single structured grid
        sheet = et.SubElement(domain, 'Grid', Name='sheet',
                GridType='Uniform')
        #Topology
        nx, ny, nz = (self.nverts_x, 1, self.nverts_z) #grid shape
        topology = et.SubElement(sheet, 'Topology', Name='topology',
                TopologyType='3DSMesh',
                Dimensions='{0} {1} {2}'.format(nx, ny, nz))
        #Geometry
        geometry = et.SubElement(sheet, 'Geometry', Name='geometry',
                GeometryType='XYZ')
        #Vertices
        m, n = self.verts.shape
        verts = et.SubElement(geometry, 'DataItem', 
                Format='XML', NumberType='Float', Precision='8',
                Dimensions='{0} {1}'.format(m,n))
        verts.text = ndarray_to_str(self.verts)
        doc.write(fn_xmf, xml_declaration=True, pretty_print=True)


    def to_obj(self, fn_obj):
        with open(fn_obj, 'w') as fh_obj:
            for i in range(self.nverts):
                write_list = ['v    '] + [str(x) for x in self.verts[i,:]]
                fh_obj.write('    '.join(write_list) + '\n')
            for i in range(self.nfaces):
                #.obj file uses 1-based indexing
                write_list = ['f    '] + [str(x+1) for x in self.faces[i,:]]
                fh_obj.write('    '.join(write_list) + '\n')


    def to_pov_mesh2(self, fn_pov):
        #povray mesh2 uses zero-based indexing
        with open(fn_pov, 'w') as fh_pov:
            fh_pov.write(str(self.nverts) + ',\n')
            for i in range(self.nverts):
                vert_as_str = ','.join([str(x) for x in self.verts[i,:]])
                pov_vec = '<' + vert_as_str + '>'
                fh_pov.write(pov_vec + ',\n')

            #Each face is broken into 2 triangles -- triA and triB
            fh_pov.write(str(2*self.nfaces) + ',\n')
            for i in range(self.nfaces):
                face_ind = self.faces[i,:]
                triA = [face_ind[0], face_ind[1], face_ind[2]]
                triB = [face_ind[0], face_ind[2], face_ind[3]]
                fh_pov.write('<' +
                       ','.join([str(x) for x in triA]) 
                        + '>' + ',\n')
                fh_pov.write('<' +
                       ','.join([str(x) for x in triB]) 
                        + '>' + ',\n')


def ndarray_to_str(A, order='C', elements_per_line=3):
    ravelA = np.ravel(A, order=order) 
    string = '\n'
    for i in range(0,len(ravelA), elements_per_line):
        buf = '  '.join([repr(x) for x in ravelA[i:i+elements_per_line]])
        buf += '\n'
        string += buf
    return string
