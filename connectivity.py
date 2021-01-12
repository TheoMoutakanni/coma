# -*- coding: utf-8 -*-
"""
Created on Wed Mar 13 17:53:30 2019

@author: Acer V-NITRO
"""

import os
import zlib
import numpy as np
import pickle
import scipy.sparse as sp


def row(A):
    return A.reshape((1, -1))

def col(A):
    return A.reshape((-1, 1))

def get_vert_connectivity(mesh):
    """Returns a sparse matrix (of size #verts x #verts) where each nonzero
    element indicates a neighborhood relation. For example, if there is a
    nonzero element in position (15,12), that means vertex 15 is connected
    by an edge to vertex 12."""

    vpv = sp.csc_matrix((len(mesh.v), len(mesh.v)))

    # for each column in the faces...
    for i in range(3):
        IS = mesh.f[:, i]
        JS = mesh.f[:, (i + 1) % 3]
        data = np.ones(len(IS))
        ij = np.vstack((row(IS.flatten()), row(JS.flatten())))
        mtx = sp.csc_matrix((data, ij), shape=vpv.shape)
        vpv = vpv + mtx + mtx.T

    return vpv

def get_vertices_per_edge(mesh, faces_per_edge=None):
    """Returns an Ex2 array of adjacencies between vertices, where
    each element in the array is a vertex index. Each edge is included
    only once. If output of get_faces_per_edge is provided, this is used to
    avoid call to get_vert_connectivity()"""

    faces = mesh.f
    suffix = str(zlib.crc32(faces_per_edge.flatten())) if faces_per_edge is not None else ''
    cache_fname = os.path.join('spring/cache/', 'verts_per_edge_cache_' + str(zlib.crc32(faces.flatten())) + '_' + suffix + '.pkl')
    
    if not os.path.exists('spring/cache/'):
        os.makedirs('spring/cache/')
    
    try:
        with open(cache_fname, 'rb') as fp:
            return(pickle.load(fp))
    except:
        if faces_per_edge is not None:
            result = np.asarray(np.vstack([row(np.intersect1d(mesh.f[k[0]], mesh.f[k[1]])) for k in faces_per_edge]), np.uint32)
        else:
            vc = sp.coo_matrix(get_vert_connectivity(mesh))
            result = np.hstack((col(vc.row), col(vc.col)))
            result = result[result[:, 0] < result[:, 1]]  # for uniqueness

        with open(cache_fname, 'wb') as fp:
            pickle.dump(result, fp, -1)
        return result

        # s1 = [set([v[0], v[1]]) for v in mesh.v]
        # s2 = [set([v[1], v[2]]) for v in mesh.v]
        # s3 = [set([v[2], v[0]]) for v in mesh.v]
        #
        # return s1+s2+s3