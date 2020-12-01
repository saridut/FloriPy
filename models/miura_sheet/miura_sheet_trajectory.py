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

from .miura_sheet import MiuraSheet


class MiuraSheetTrajectory(object):
    def __init__(self, fn_traj, fn_model):
        self._fh_traj = h5py.File(fn_traj, 'r')
        self._modelspec = yamlio.read(fn_model)


    def close(self):
        self._fh_traj.close()


    def __len__(self):
        return len(self._fh_traj['ts'])


    def get_frame(self, i):
        grp = self._fh_traj['ts/{0}'.format(i)]
        time = float(grp.attrs['time']) #HDF5 scalar
        df_traj = {'vertices': grp['vertices'][...], 'com': grp['com'][...]}
        ms = MiuraSheet.from_df_traj(df_traj, self._modelspec)
        return (time, ms)


    @classmethod
    def create(cls, fn_traj, fn_model):
        modelspec = yamlio.read(fn_model)
        fh_traj = h5py.File(fn_traj, 'r+')
        for key in fh_traj['ts']:
            grp = fh_traj['ts/'+key]
            if 'vertices' not in grp:
                body_coms = grp['body_coms'][...]
                body_oris = grp['body_orientations'][...]
                df_mbs = {'body_coms': body_coms,
                        'body_orientations': body_oris}
                ms = MiuraSheet.from_df_mbs(df_mbs, modelspec)
                verts = grp.create_dataset('vertices', data=ms.verts)
                grid_shape = np.array([ms.nverts_x, 1, ms.nverts_z], dtype=np.int32)
                verts.attrs['grid_shape'] = grid_shape
                com = grp.create_dataset('com', data=ms.com)
        fh_traj.close()
        mst = cls(fn_traj, fn_model)
        return mst


def create_xdmf(fn_h5, fn_xmf):
    '''
    Create an xdmf file from HDF5 trajectory file.

    '''
    fh_h5 = h5py.File(fn_h5, 'r')
    xdmf = et.XML('''<?xml version="1.0"?>
            <!DOCTYPE Xdmf SYSTEM "Xdmf.dtd" []>
            <Xdmf Version="3.0" xmlns:xi="http://www.w3.org/2001/XInclude"/>
            ''')
    doc = et.ElementTree(element=xdmf)
    domain = et.SubElement(xdmf, 'Domain')
    traj = et.SubElement(domain, 'Grid', Name='traj', GridType='Collection',
                CollectionType='Temporal')
    grp = fh_h5['ts']
    for iframe in range(len(grp)):
        key = str(iframe)
        m, n = grp[key]['vertices'].shape
        nx, ny, nz = tuple(grp[key]['vertices'].attrs['grid_shape'])
        sheet = et.SubElement(traj, 'Grid', Name='sheet', GridType='Uniform')
        time = et.SubElement(sheet, 'Time', Value=str(grp[key].attrs['time']))
        topology = et.SubElement(sheet, 'Topology', Name='topology',
                TopologyType='3DSMesh',
                Dimensions='{0} {1} {2}'.format(nx,ny,nz))
        geometry = et.SubElement(sheet, 'Geometry', Name='geometry',
                GeometryType='XYZ')
        verts = et.SubElement(geometry, 'DataItem', Name='vertices',
                Format='HDF', NumberType='Float', Precision='8',
                Dimensions='{0} {1}'.format(m,n))
        verts.text = fh_h5.filename + ':' + grp[key]['vertices'].name
    doc.write(fn_xmf, xml_declaration=True, pretty_print=True) 
    fh_h5.close()
