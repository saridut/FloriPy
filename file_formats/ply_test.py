#!/usr/bin/env python

import numpy as np
import plyfile

vertices = np.array([(0, 0, 0),
                     (1, 0, 0), 
                     (1, 1, 0),
                     (0, 1, 0),
                     (2, 0, 0),
                     (2, 1, 0)],
                     dtype=[('x',np.float64), ('y',np.float64),
                         ('z',np.float64)])

faces = np.array([([0,1,2,3],),
                  ([1,4,5,2],)],
                  dtype=[('vertex_indices', np.int32, (4,))])
#For polygons with different number of vertices
#                 dtype=[('vertex_indices', 'O')])

vertex_element = plyfile.PlyElement.describe(vertices, 'vertex')
face_element = plyfile.PlyElement.describe(faces, 'face')

plyfile.PlyData([vertex_element, face_element],
        text=False, comments=['Test ply writer']).write('binary_out.ply')
