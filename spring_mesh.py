# -*- coding: utf-8 -*-
"""
Created on Sun Mar 10 17:13:50 2019

@author: Acer V-NITRO
"""

from sklearn.decomposition import PCA
import vtk
from vtk.numpy_interface import dataset_adapter as dsa

import glob
from tqdm import tqdm

import numpy as np

from mesh import Mesh

if mats is None:
    mats = []
    for filename in tqdm(glob.glob('./data/spring/SPRING_FEMALE/*.obj')):
        reader = vtk.vtkOBJReader()
        reader.SetFileName(filename)
        reader.Update()
        
        np_mesh = dsa.WrapDataObject(reader.GetOutput())
        verts = np_mesh.GetPoints()
        mats.append(verts)

reader = vtk.vtkOBJReader()
reader.SetFileName('./data/spring/SPRING_FEMALE/SPRING0014.obj')
reader.Update()
np_mesh = dsa.WrapDataObject(reader.GetOutput())
faces = np_mesh.GetPolygons()
faces = faces.reshape((-1, 4))[:, 1:]

n_components=50

mats = np.array(mats).reshape((-1,12500*3))

pca = PCA(n_components=n_components)
pca.fit(mats)
print(['{:2f}'.format(v) for v in pca.explained_variance_ratio_])

std = np.sqrt(pca.explained_variance_)
eigenshapes = pca.components_
mean_mats = pca.mean_

component_p = []
component_m = []
for id in range(6):
    #component_p.append((mean_mats + 3*std[id]*eigenshapes[id]).reshape((-1,3)))
    #component_m.append((mean_mats - 3*std[id]*eigenshapes[id]).reshape((-1,3)))
    component_p.append(pca.inverse_transform(pca.transform([mats[id]]))[0].reshape((-1,3)))
    component_m.append(mats[id].reshape((-1,3)))


def create_polydata(points, faces):
    vertices = vtk.vtkPoints()
    
    for p in points:
        vertices.InsertNextPoint(p)
    
    triangles = vtk.vtkCellArray()
    
    for (p1,p2,p3) in faces:
        triangle = vtk.vtkTriangle()
        
        triangle.GetPointIds().SetId ( 0, p1);
        triangle.GetPointIds().SetId ( 1, p2);
        triangle.GetPointIds().SetId ( 2, p3);
        
        triangles.InsertNextCell(triangle)
    
    polydata = vtk.vtkPolyData()
    
    polydata.SetPoints(vertices)
    polydata.SetPolys(triangles)
    return polydata

mean_shape = Mesh(v=mean_mats.reshape((-1,3)),f=faces)
mean_shape.write_obj("./data/template.obj")

renderWindow = vtk.vtkRenderWindow()
renderWindowInteractor = vtk.vtkRenderWindowInteractor()
renderWindowInteractor.SetRenderWindow(renderWindow)

N,M = 2,3
xmins=[i/N for i in range(0,N)]
xmaxs=[i/N for i in range(1,N+1)]
ymins=[i/M for i in np.array(list(zip(*[range(M)]*N))).flatten()]
ymaxs=[i/M for i in np.array(list(zip(*[range(1,M+1)]*N))).flatten()]

renderers = []
for i in range(M*N):
    renderers.append(vtk.vtkRenderer())
    
for i in range(M*N):
    renderWindow.AddRenderer(renderers[i])
    renderers[i].SetViewport(xmins[i%N],ymins[i],xmaxs[i%N],ymaxs[i])

def render(polydata, i, color=(1,1,1), opacity=1.):
    renderer = renderers[i]
    mapper = vtk.vtkPolyDataMapper()
    mapper.SetInputData(polydata)
    actor = vtk.vtkActor()
    actor.SetMapper(mapper)
    actor.GetProperty().SetColor(*color)
    actor.GetProperty().SetOpacity(opacity)
    renderer.AddActor(actor)

d_x,d_y = 0.25,-0.4

for id in range(6):
    render(create_polydata(component_p[id]+(d_x,d_y,0),faces), id, (1,1,1))
    render(create_polydata(component_m[id]-(d_x,d_y,0),faces), id, (1,1,1))

renderWindowInteractor.Initialize()
renderWindow.Render()
renderWindowInteractor.Start()