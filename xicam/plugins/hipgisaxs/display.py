import pyqtgraph as pg
import pyqtgraph.opengl as gl
from OpenGL.GL import *
import latvec
import numpy as np
from PySide import QtGui
import featuremanager
import customwidgets

viewWidget = None

boxsize = 100

def load():
    global viewWidget
    viewWidget = orthoGLViewWidget()
    redraw()


class orthoGLViewWidget(gl.GLViewWidget):
    def __init__(self):
        super(orthoGLViewWidget, self).__init__()
        self.opts['distance']=1500

    def projectionMatrix(self, region=None):
        # Xw = (Xnd + 1) * width/2 + X
        if region is None:
            region = (0, 0, self.width(), self.height())

        x0, y0, w, h = self.getViewport()
        dist = self.opts['distance']
        fov = self.opts['fov']
        nearClip = dist * 0.0001
        farClip = dist * 1000.

        r = nearClip * np.tan(fov * 0.5 * np.pi / 180.)
        t = r * h / w

        # convert screen coordinates (region) to normalized device coordinates
        # Xnd = (Xw - X0) * 2/width - 1
        ## Note that X0 and width in these equations must be the values used in viewport
        left = r * ((region[0] - x0) * (2.0 / w) - 1)
        right = r * ((region[0] + region[2] - x0) * (2.0 / w) - 1)
        bottom = t * ((region[1] - y0) * (2.0 / h) - 1)
        top = t * ((region[1] + region[3] - y0) * (2.0 / h) - 1)

        tr = QtGui.QMatrix4x4()
        tr.ortho(left * 2000, right * 2000, bottom * 2000, top * 2000, -10, farClip)
        return tr


class latticeFrame(gl.GLGridItem):
    def __init__(self, lines, antialias=True, glOptions='translucent'):
        super(latticeFrame, self).__init__(size=None, color=None)#, antialias=True, glOptions='translucent')
        self.setGLOptions(glOptions)
        self.antialias = antialias
        self.lines = lines
       # print lines


    def paint(self):
        # print 'painting!'
        self.setupGLState()
        if self.antialias:
            glEnable(GL_LINE_SMOOTH)
            glEnable(GL_BLEND)
            glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
            glHint(GL_LINE_SMOOTH_HINT, GL_NICEST)

        # glEnable(GL_LINE_SMOOTH)
        # glEnable(GL_BLEND)
        # glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
        # glHint(GL_LINE_SMOOTH_HINT, GL_NICEST)
        #
        glLineWidth(3)
        #glColor4f(1, .5, 0., .65)
        glColor4f(0,1.,1.,.65)

        glBegin(GL_LINES)
        for line in self.lines:
            # print 'line:',line
            for vertex in line:
                glVertex3fv(vertex)
                # print 'vertex:', vertex
        glEnd()
        glLineWidth(1)


def clear():

    for item in viewWidget.items:
        item._setView(None)
    grid = gl.GLGridItem(size=QtGui.QVector3D(200,200,0))
    grid.setSpacing(10,10,10)
    viewWidget.items = [grid]
    addLayer(-20., 0.)
    viewWidget.update()


def showLattice(a, b, c, orders=7, basis=None, shape='Sphere', z0=0, xrot=0, yrot=0, zrot=0, **kwargs):
    vecs = latvec.latticevectors(a, b, c, kwargs['radius'], maxreps=100, repetitions = kwargs['repetitions'], scaling=kwargs['scaling'])

    linez=0

    if basis is None:
        basis = [(0, 0, 0)]

    for basisvec in basis:
        for vec in vecs:
            if shape=='Sphere':
                addSphere(np.sum([map(np.add,vec,[0,0,z0]), basisvec], axis=0),[kwargs['radius']]*3,xrot,yrot,zrot)
                linez=kwargs['radius']
            elif shape=='Box':
                addBox(np.sum([map(np.add,vec,[0,0,z0]), basisvec], axis=0),[kwargs['length'],kwargs['width'],kwargs['height']],xrot,yrot,zrot)
                linez=kwargs['height']
            elif shape == 'Cylinder':
                addCylinder(np.sum([map(np.add, vec, [0, 0, z0]), basisvec], axis=0),
                       [kwargs['radius'], kwargs['radius'], kwargs['height']],xrot,yrot,zrot)
                linez = kwargs['height']


    lines = latvec.latticelines(a, b, c, linez+z0, maxreps=100, repetitions=kwargs['repetitions'], scaling=kwargs['scaling'])

    viewWidget.addItem(latticeFrame(lines))

def addBox(center, scale, xrot, yrot, zrot):
    verts = np.array([[-.5, -.5, .5],
                      [-.5, .5, .5],
                      [.5, -.5, .5],
                      [.5, .5, .5],
                      [-.5, -.5, -.5],
                      [-.5, .5, -.5],
                      [.5, -.5, -.5],
                      [.5, .5, -.5]])
    faces = np.array(
        [[0, 2, 1], [1, 2, 3], [0, 1, 4], [1, 5, 4], [1, 3, 5], [3, 7, 5], [3, 2, 7], [2, 6, 7], [2, 0, 6], [0, 4, 6],
         [4, 5, 6], [5, 7, 6]])

    box = gl.GLMeshItem(vertexes=verts,faces = faces, color=(1,0,1,.3), shader = 'shaded', smooth=True, glOptions='opaque')
    box.rotate(xrot, 1, 0, 0)
    box.rotate(yrot, 0, 1, 0)
    box.rotate(zrot, 0, 0, 1)
    box.translate(*center)
    box.scale(*scale)
    viewWidget.addItem(box)


def cylinder(rows, cols, radius=[1.0, 1.0], length=1.0, offset=False):
    """
    Return a MeshData instance with vertexes and faces computed
    for a cylindrical surface.
    The cylinder may be tapered with different radii at each end (truncated cone)
    """
    verts = np.empty((rows + 1, cols, 3), dtype=float)
    if isinstance(radius, int):
        radius = [radius, radius]  # convert to list
    ## compute vertexes
    th = np.linspace(2 * np.pi, 0, cols).reshape(1, cols)
    r = np.linspace(radius[0], radius[1], num=rows + 1, endpoint=True).reshape(rows + 1, 1)  # radius as a function of z
    verts[..., 2] = np.linspace(0, length, num=rows + 1, endpoint=True).reshape(rows + 1, 1)  # z
    if offset:
        th = th + ((np.pi / cols) * np.arange(rows + 1).reshape(rows + 1, 1))  ## rotate each row by 1/2 column
    verts[..., 0] = r * np.cos(th)  # x = r cos(th)
    verts[..., 1] = r * np.sin(th)  # y = r sin(th)
    verts = verts.reshape((rows + 1) * cols, 3)  # just reshape: no redundant vertices...
    ## compute faces
    faces = np.empty((rows * cols * 2, 3), dtype=np.uint)
    rowtemplate1 = ((np.arange(cols).reshape(cols, 1) + np.array([[0, 1, 0]])) % cols) + np.array([[0, 0, cols]])
    rowtemplate2 = ((np.arange(cols).reshape(cols, 1) + np.array([[0, 1, 1]])) % cols) + np.array([[cols, 0, cols]])
    for row in range(rows):
        start = row * cols * 2
        faces[start:start + cols] = rowtemplate1 + row * cols
        faces[start + cols:start + (cols * 2)] = rowtemplate2 + row * cols

    verts=np.vstack([verts,[0,0,0],[0,0,1]])
    top = ((np.arange(cols).reshape(cols, 1) + np.array([[0, 1, 0]])) % cols) * np.array([1,1,0]) + np.array([[0, 0, len(verts)-2]])
    down = ((np.arange(cols).reshape(cols, 1) + np.array([[0, 1, 1]])) % cols) * np.array([1,1,0]) + np.array([[cols, cols, len(verts)-1]])

    faces=np.vstack([faces,top,down])


    return gl.MeshData(vertexes=verts, faces=faces.astype(np.int))

def addCylinder(center, scale, xrot, yrot, zrot):
    md = cylinder(rows=1, cols=20)
    alpha = .3
    cyl = gl.GLMeshItem(meshdata=md, smooth=True, color=(1, 0, 1, alpha), shader='shaded')
    cyl.translate(0, 0, -scale[2] / 2.)
    cyl.rotate(xrot, 1, 0, 0)
    cyl.rotate(yrot, 0, 1, 0)
    cyl.rotate(zrot, 0, 0, 1)
    cyl.translate(*center)
    cyl.scale(*scale)

    viewWidget.addItem(cyl)

def addSphere(center, scale, xrot, yrot, zrot):
    md = gl.MeshData.sphere(rows=5, cols=10)
    alpha = .3
    sphere = gl.GLMeshItem(meshdata=md, smooth=True, color=(1, 0, 1, alpha), shader='shaded')
    # No rotation for spheres!
    sphere.translate(*center)
    sphere.scale(*scale)
    viewWidget.addItem(sphere)


def addLayer(z0, z1, glOptions='opaque'):
    verts = np.array([[-boxsize, -boxsize, z0],
                      [-boxsize, boxsize, z0],
                      [boxsize, -boxsize, z0],
                      [boxsize, boxsize, z0],
                      [-boxsize, -boxsize, z1],
                      [-boxsize, boxsize, z1],
                      [boxsize, -boxsize, z1],
                      [boxsize, boxsize, z1]])

    faces = np.array(
        [[0, 2, 1], [1, 2, 3], [0, 1, 4], [1, 5, 4], [1, 3, 5], [3, 7, 5], [3, 2, 7], [2, 6, 7], [2, 0, 6], [0, 4, 6],
         [4, 5, 6], [5, 7, 6]])

    layer = gl.GLMeshItem(vertexes=verts, faces=faces, color=(.3, .3, .3, .8), smooth=True, shader='shaded',
                          glOptions=glOptions)
    viewWidget.addItem(layer)

def redraw(*args,**kwargs):
    clear()
    layerz=0
    particlez=0
    for feature in featuremanager.features:
        if type(feature) is customwidgets.layer:
            particlez=layerz
            addLayer(layerz,layerz+feature.Thickness.value()*1.e9,glOptions='additive')
            layerz+=feature.Thickness.value()*1.e9
        elif type(feature) is customwidgets.particle:
            basis = [vecparam.value() for vecparam in feature.structure.Basis.children()]
            showLattice(map(float, feature.structure.LatticeA.value()), map(float, feature.structure.LatticeB.value()),
                            map(float, feature.structure.LatticeC.value()), basis=basis, z0=particlez,
                        shape=feature.Type.value(), radius=feature.Radius.Value.value(), height=feature.Height.Value.value(),
                        width=feature.Width.Value.value(), length=feature.Length.Value.value(), baseangle=feature.BaseAngle.Value.value(),
                        xrot=feature.XRotation.value(),yrot=feature.YRotation.value(),zrot=feature.ZRotation.value(),
                        repetitions=feature.structure.Repetition.value(),scaling=feature.structure.Scaling.value())