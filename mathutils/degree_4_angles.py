#!/usr/bin/env python

import numpy as np
from numpy import linalg as la

vertices_xy = np.array([
                        [-2.0, -2.0],
                        [ 0.0, -2.0],
                        [ 2.0, -2.0],
                        [-2.0, -0.2],
                        [ 0.0,  0.0],
                        [ 2.0, -0.2],
                        [-2.0,  2.0],
                        [ 0.0,  2.0],
                        [ 2.0,  2.0]
                        ])

fold_edges = [(4, 1), (4, 5), (4, 7), (4, 3)]
num_fold_edges = 4
fold_edge_vectors = np.zeros((num_fold_edges, 2))

for i in range(num_fold_edges):
    u, v = fold_edges[i]
    edge_vector = (vertices_xy[v]-vertices_xy[u])/la.norm(vertices_xy[v]
                        -vertices_xy[u])
    fold_edge_vectors[i,:] = edge_vector

alpha = np.zeros((num_fold_edges,))
alpha[0] = np.arccos(np.dot(fold_edge_vectors[1], fold_edge_vectors[0]))
alpha[1] = np.arccos(np.dot(fold_edge_vectors[2], fold_edge_vectors[1]))
alpha[2] = np.arccos(np.dot(fold_edge_vectors[3], fold_edge_vectors[2]))
alpha[3] = np.arccos(np.dot(fold_edge_vectors[0], fold_edge_vectors[3]))

gamma = np.zeros((num_fold_edges,))
gamma[0] = np.deg2rad(-45.0)
gamma[2] = -gamma[0]
gamma[1] = 2*np.arctan(np.tan(gamma[0]/2)*np.sin((alpha[0]+alpha[1])/2)/np.sin((alpha[0]
                    -alpha[1])/2))
gamma[3] = gamma[1]

for i in range(num_fold_edges):
    print('alpha ', i, np.rad2deg(alpha[i]))
for i in range(num_fold_edges):
    print('gamma ', i, repr(np.rad2deg(gamma[i])))

