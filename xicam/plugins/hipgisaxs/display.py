import pyqtgraph as pg
import pyqtgraph.opengl as gl
from OpenGL.GL import *
import latvec
import numpy as np
from PySide import QtGui

viewWidget = None
grid = None

boxsize = 10
spherescale =.2


def load():
    global viewWidget, grid
    viewWidget = orthoGLViewWidget()

    grid = gl.GLGridItem()
    addLayer(-1., 0.)

    viewWidget.addItem(grid)


class orthoGLViewWidget(gl.GLViewWidget):
    def projectionMatrix(self, region=None):
        # Xw = (Xnd + 1) * width/2 + X
        if region is None:
            region = (0, 0, self.width(), self.height())

        x0, y0, w, h = self.getViewport()
        dist = self.opts['distance']
        fov = self.opts['fov']
        nearClip = dist * 0.001
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
        glColor4f(1, .5, 0., .65)

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
        if not item == grid:
            item._setView(None)

    viewWidget.items = [grid]
    addLayer(-1., 0.)
    viewWidget.update()


def showLattice(a, b, c, orders=7, basis=None, zoffset=spherescale):
    clear()

    vecs = latvec.latticevectors(a, b, c, zoffset, orders)

    if basis is None:
        basis = [(0, 0, 0)]

    for basisvec in basis:
        for vec in vecs:
            addSphere(np.sum([vec, basisvec], axis=0),[spherescale]*3)

    lines = latvec.latticelines(a, b, c, zoffset, orders-1)

    viewWidget.addItem(latticeFrame(lines))


def addSphere(center, scale):
    md = gl.MeshData.sphere(rows=5, cols=10)
    alpha = .3
    sphere = gl.GLMeshItem(meshdata=md, smooth=True, color=(0, 1, 1, alpha), shader='shaded')
    sphere.translate(*center)
    sphere.scale(*scale)
    viewWidget.addItem(sphere)


def addLayer(z0, z1):
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
                          glOptions='opaque')
    viewWidget.addItem(layer)
