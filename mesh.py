# -*- coding: utf-8 -*-
"""
Created on Tue Mar 12 22:58:57 2019

@author: Acer V-NITRO
"""

import numpy as np
import scipy.io as sio
import os
import re

class Mesh(object):
    """3d Triangulated Mesh class
    Attributes:
        v: Vx3 array of vertices
        f: Fx3 array of faces
    """
    def __init__(self,
                 v=None,
                 f=None,
                 filename=None,
                 basename=None):
        """
        :param v: vertices
        :param f: faces
        :param filename: a filename from which a mesh is loaded
        """

        if filename is not None:
            self.load_from_file(filename)
            if hasattr(self, 'f'):
                self.f = np.require(self.f, dtype=np.uint32)
            self.v = np.require(self.v, dtype=np.float64)
            self.filename = filename
        if v is not None:
            self.v = np.array(v, dtype=np.float64)
        if f is not None:
            self.f = np.require(f, dtype=np.uint32)

        self.basename = basename
        if self.basename is None and filename is not None:
            self.basename = os.path.splitext(os.path.basename(filename))[0]

    def show(self, mv=None, meshes=[], lines=[]):
        from .meshviewer import MeshViewer
        from .utils import row

        if mv is None:
            mv = MeshViewer(keepalive=True)

        if hasattr(self, 'landm'):
            from .sphere import Sphere
            sphere = Sphere(np.zeros((3)), 1.).to_mesh()
            scalefactor = 1e-2 * np.max(np.max(self.v) - np.min(self.v)) / np.max(np.max(sphere.v) - np.min(sphere.v))
            sphere.v = sphere.v * scalefactor
            spheres = [Mesh(vc='SteelBlue', f=sphere.f, v=sphere.v + row(np.array(self.landm_raw_xyz[k]))) for k in self.landm.keys()]
            mv.set_dynamic_meshes([self] + spheres + meshes, blocking=True)
        else:
            mv.set_dynamic_meshes([self] + meshes, blocking=True)
        mv.set_dynamic_lines(lines)
        return mv

    def faces_by_vertex(self, as_sparse_matrix=False):
        import scipy.sparse as sp
        if not as_sparse_matrix:
            faces_by_vertex = [[] for i in range(len(self.v))]
            for i, face in enumerate(self.f):
                faces_by_vertex[face[0]].append(i)
                faces_by_vertex[face[1]].append(i)
                faces_by_vertex[face[2]].append(i)
        else:
            row = self.f.flatten()
            col = np.array([range(self.f.shape[0])] * 3).T.flatten()
            data = np.ones(len(col))
            faces_by_vertex = sp.csr_matrix((data, (row, col)), shape=(self.v.shape[0], self.f.shape[0]))
        return faces_by_vertex

    def estimate_vertex_normals(self, face_to_verts_sparse_matrix=None):
        from .geometry.tri_normals import TriNormalsScaled

        face_normals = TriNormalsScaled(self.v, self.f).reshape(-1, 3)
        ftov = face_to_verts_sparse_matrix if face_to_verts_sparse_matrix else self.faces_by_vertex(as_sparse_matrix=True)
        non_scaled_normals = ftov * face_normals
        norms = (np.sum(non_scaled_normals ** 2.0, axis=1) ** 0.5).T
        norms[norms == 0] = 1.0
        return (non_scaled_normals.T / norms).T

    def barycentric_coordinates_for_points(self, points, face_indices):
        from .geometry.barycentric_coordinates_of_projection import barycentric_coordinates_of_projection
        vertex_indices = self.f[face_indices.flatten(), :]
        tri_vertices = np.array([self.v[vertex_indices[:, 0]], self.v[vertex_indices[:, 1]], self.v[vertex_indices[:, 2]]])
        return vertex_indices, barycentric_coordinates_of_projection(points, tri_vertices[0, :], tri_vertices[1, :] - tri_vertices[0, :], tri_vertices[2, :] - tri_vertices[0, :])

    def load_from_file(self, filename):
        if re.search(".ply$", filename):
            self.load_from_ply(filename)
        elif re.search(".obj$", filename):
            self.load_from_obj(filename)
        elif re.search(".mat$", filename):
            self.load_from_mat(filename)
        else:
            raise NotImplementedError("Unknown mesh file format.")

    def load_from_ply(self, filename):
        raise("can't load from ply")
    
    def load_from_mat(self, filename):
        self.v = sio.loadmat(filename)['points']
        
        faces = []
        with open(os.path.join(os.path.dirname(os.path.abspath(filename)),'faces.txt'),'r') as f:
            lines = f.readlines()
            for i,line in enumerate(lines):
                if i != len(lines)-1:
                    l = line.split(' ')[:-1]
                faces.append(list(map(int,l)))
        self.f = faces

    def write_json(self, filename, header="", footer="", name="", include_faces=True, texture_mode=True):
        raise("can't write json")

    def write_three_json(self, filename, name=""):
        raise("can't write three json")

    def write_ply(self, filename, flip_faces=False, ascii=False, little_endian=True, comments=[]):
        raise("can't write ply")

    def write_mtl(self, path, material_name, texture_name):
        """Serializes a material attributes file"""
        raise("can't write mtl")

    def load_from_obj(self, filename):
        v = []
        f = []
        fn = []
        vn = []
        currLandm = ''
        with open(filename, 'r', buffering=2 ** 10) as fp:
            for line in fp:
                line = line.split()
                if len(line) > 0:
                    if line[0] == 'v':
                        v.append([float(x) for x in line[1:4]])
                        if currLandm:
                            currLandm = ''
                    elif line[0] == 'vn':
                        vn.append([float(x) for x in line[1:]])
                    elif line[0] == 'f':
                        faces = [x.split('/') for x in line[1:]]
                        for iV in range(1, len(faces) - 1):  # trivially triangulate faces
                            f.append([int(faces[0][0]), int(faces[iV][0]), int(faces[iV + 1][0])])
                            if (len(faces[0]) > 2) and faces[0][2]:
                                fn.append([int(faces[0][2]), int(faces[iV][2]), int(faces[iV + 1][2])])
    
        self.v = np.array(v)
        self.f = np.array(f) - 1
        if vn:
            self.vn = np.array(vn)
        if fn:
            self.fn = np.array(fn) - 1
    
    
    def write_obj(self, filename, flip_faces=False, group=False, comments=None):
        if os.path.dirname(filename) and not os.path.exists(os.path.dirname(filename)):
            os.makedirs(os.path.dirname(filename))
    
        ff = -1 if flip_faces else 1
    
        def write_face_to_obj_file(face_index, obj_file):
            vertex_indices = self.f[face_index][::ff] + 1
    
            if hasattr(self, 'fn'):
                normal_indices = self.fn[face_index][::ff] + 1
                obj_file.write('f %d//%d %d//%d  %d//%d\n' % tuple(np.array([vertex_indices, normal_indices]).T.flatten()))
            else:
                obj_file.write('f %d %d %d\n' % tuple(vertex_indices))
    
        with open(filename, 'w') as fi:
            for r in self.v:
                fi.write('v %f %f %f\n' % (r[0], r[1], r[2]))
    
            if hasattr(self, 'fn') and hasattr(self, 'vn'):
                for r in self.vn:
                    fi.write('vn %f %f %f\n' % (r[0], r[1], r[2]))
    
            if hasattr(self, 'f'):
                for face_index in range(len(self.f)):
                    write_face_to_obj_file(face_index, fi)