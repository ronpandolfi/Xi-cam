import os
import numpy as np
import vtk
from vtk.util import numpy_support
import base
import pyqtgraph.opengl as gl
from pipeline import loader

def polytopointstriangles(polys):
    vertexes = np.array([polys.GetPoint(i) for i in xrange(polys.GetNumberOfPoints())],dtype=int)

    pdata = polys.GetPolys().GetData()

    values = [int(pdata.GetTuple1(i)) for i in xrange(pdata.GetNumberOfTuples())]

    triangles = []
    while values:
        n = values[0]       # number of points in the polygon
        triangles.append(values[1:n+1])
        del values[0:n+1]

    return vertexes, triangles

def numpytovtk(dataarray):
    dataarray=np.ascontiguousarray(dataarray)
    VTK_data = numpy_support.numpy_to_vtk(num_array=dataarray, deep=True, array_type=vtk.VTK_FLOAT)
    return VTK_data

def loadafmasmesh(path, flatten=True, gaussianblursize=5):

    img = loader.loadimage(path)
    print img

    #make a grid array for x and y
    xx, yy = np.mgrid[0:img.shape[0], 0:img.shape[1]]

    #make the array of points, then reshape it to an Nx3 array
    points = np.array([xx, yy, img])
    points = points.reshape((3, img.shape[0] * img.shape[1])).T

    pointsforvtk = vtk.vtkPoints()
    polygondata=vtk.vtkPolyData()
    cellarray=vtk.vtkCellArray()
    delaunay=vtk.vtkDelaunay2D()
    boundary=vtk.vtkPolyData()

    pointsarray = numpytovtk(points)
    pointsforvtk.SetData(pointsarray)
    polygondata.SetPoints(pointsforvtk)
    boundary.SetPoints(polygondata.GetPoints())
    boundary.SetPolys(cellarray)

    delaunay.SetInputData(polygondata)
    delaunay.SetSourceData(boundary)

    decimator = vtk.vtkDecimatePro()
    decimator.SetInputConnection(delaunay.GetOutputPort())
    decimator.SetTargetReduction(0.99)
    decimator.PreserveTopologyOn()

    decimator.Update()

    print "mesh finished"
    points,triangles=polytopointstriangles(decimator.GetOutput())
    print(triangles.__len__())

    return points, triangles

def writeobj(vertices,triangles,filename):
    with open(filename,'w') as f:
        for vertex in vertices:
            f.write('v {0} {1} {2}\n'.format(vertex[0],vertex[1],vertex[2]))
        for triangle in triangles:
            f.write('f {0} {1} {2}\n'.format(triangle[0]+1,triangle[1]+1,triangle[2]+1))

def openfiles(filepaths):
    pixelsize = Triangulator.parameters.child('pixelsize').value()
    for path in filepaths:
        points, triangles = loadafmasmesh(path)
        displaypoints = points / pixelsize
        displaypoints[:,2]*=10

        mesh = gl.GLMeshItem(vertexes=displaypoints,
                             faces=triangles,
                             smooth=False,
                             shader='shaded',
                             glOptions='opaque',
                             drawEdges=True,
                             edgeColor=(1,1,1,1))

        for item in Triangulator.centerwidget.items:
            Triangulator.centerwidget.removeItem(item)

        dx,dy,dz=0,0,0
        # dx=-img.shape[0]/2
        # dy=-img.shape[1]/2
        # dz=-np.average(img)
        mesh.translate(dx,dy,dz,local=True)
        Triangulator.centerwidget.addItem(mesh)
        print('awesome')

    writeobj(points, triangles, os.path.splitext(path)[0] + '.obj')

Triangulator = base.EZplugin(name='Triangulator',
                             parameters=[{'name': 'gaussian', 'value': 5, 'type': 'int'},
                                         {'name': 'pixelsize', 'value': 1, 'type': 'int'},
                                         {'name': 'Target decimation', 'value': 1, 'type': 'int'},
                                         {'name': 'Display scale', 'value': 1, 'type': 'int'}],
                             openfileshandler=openfiles,
                             centerwidget=gl.GLViewWidget,
                             bottomwidget=lambda: None)