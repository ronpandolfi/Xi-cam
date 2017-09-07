import base
import numpy as np
import numpy.matlib
from scipy.fftpack import *
import scipy.ndimage
from numpy.fft import *
from afmmesher import loadafmasmesh

def runtest():
    reso = 18
    pitch = EZTest.parameters.child('Pitch').value()
    r = EZTest.parameters.child('Radius').value()
    shell = EZTest.parameters.child('Shell').value()
    fact_rs = EZTest.parameters.child('Coherence').value()

    for angl in range(0, 1, 1):
        angle = angl

        # Distance between pillar
        width = np.int((pitch * np.cos(np.radians(angle))) / reso)
        height = np.int(pitch / reso)

        # radius of the pillar
        radiush = r / reso
        radiusl = (r * np.cos(np.radians(angle))) / reso

        radiush1 = (r - shell) / reso
        radiusl1 = ((r - shell) * np.cos(np.radians(angle))) / reso

        centerW = width / 2
        centerH = height / 2

        W = np.linspace(1, width, width)
        H = np.linspace(1, height, height)

        mask = np.zeros((width, height))

        for i in range(0, width, 1):
            for j in range(0, height, 1):
                if ((W[i] - centerW) / radiusl) ** 2 + ((H[j] - centerH) / radiush) ** 2 < 1:
                    mask[i, j] = 10
                if ((W[i] - centerW) / radiusl1) ** 2 + ((H[j] - centerH) / radiush1) ** 2 < 1:
                    mask[i, j] = 0

        B = np.zeros((fact_rs * width, fact_rs * height))
        Ifft = np.zeros((fact_rs * width, fact_rs * height))

        B = np.matlib.repmat(mask, fact_rs, fact_rs)

    EZTest.setImage(B)
    np.save('C:\\Users\\tpicardjoly\\Desktop\\img3d.npy', B)


def img3d():
    img = EZTest.centerwidget.image
    points, triangles = loadafmasmesh('C:\\Users\\tpicardjoly\\Desktop\\img3d.npy')
    np.save('C:\\Users\\tpicardjoly\\Desktop\\img3d.npy', triangles)

def runtest2():
    reso = 18
    # Distance between pillar
    pitch = np.double(EZTest.parameters.child('Pitch').value())
    r = np.double(EZTest.parameters.child('Radius').value())
    shell = np.double(EZTest.parameters.child('Shell').value())
    fact_rs = EZTest.parameters.child('Coherence').value()
    det = EZTest.parameters.child('Detector').value()

    for angl in range(0, 1, 1):
        angle = angl

        width = np.int((pitch * np.cos(np.radians(angle))) / reso)
        height = np.int(pitch / reso)

        # radius of the pillar
        radiush = r / reso
        radiusl = (r * np.cos(np.radians(angle))) / reso

        radiush1 = (r - shell) / reso
        radiusl1 = ((r - shell) * np.cos(np.radians(angle))) / reso

        centerW = width / 2
        centerH = height / 2

        W = np.linspace(1, width, width)
        H = np.linspace(1, height, height)

        mask = np.zeros((width, height))

        for i in range(0, width, 1):
            for j in range(0, height, 1):
                if ((W[i] - centerW) / radiusl) ** 2 + ((H[j] - centerH) / radiush) ** 2 < 1:
                    mask[i, j] = 10
                if ((W[i] - centerW) / radiusl1) ** 2 + ((H[j] - centerH) / radiush1) ** 2 < 1:
                    mask[i, j] = 0
                    # fact_rs = 3
        B = np.zeros((fact_rs * width, fact_rs * height))
        Ifft = np.zeros((fact_rs * width, fact_rs * height))

        B = np.matlib.repmat(mask, fact_rs, fact_rs)
        Ifft = np.abs(fftshift(fftn(B)))

        sc = det / len(B);
        J = np.double(scipy.misc.imresize(Ifft, 100 * sc, 'bilinear'))
        J = J ** 2

    img = np.log(0.01 + J)
    EZTest.setImage(img)


def run_change():
    reso = 18
    # Distance between pillar
    pitch = np.double(EZTest.parameters.child('Pitch').value())
    r = np.double(EZTest.parameters.child('Radius').value())
    shell = np.double(EZTest.parameters.child('Shell').value())
    fact_rs = EZTest.parameters.child('Coherence').value()
    det = EZTest.parameters.child('Detector').value()

    for angl in range(0, 1, 1):
        angle = angl

        width = np.int((pitch * np.cos(np.radians(angle))) / reso)
        height = np.int(pitch / reso)

        # radius of the pillar
        radiush = r / reso
        radiusl = (r * np.cos(np.radians(angle))) / reso

        radiush1 = (r - shell) / reso
        radiusl1 = ((r - shell) * np.cos(np.radians(angle))) / reso

        centerW = width / 2
        centerH = height / 2

        W = np.linspace(1, width, width)
        H = np.linspace(1, height, height)

        mask = np.zeros((width, height))

        for i in range(0, width, 1):
            for j in range(0, height, 1):
                if ((W[i] - centerW) / radiusl) ** 2 + ((H[j] - centerH) / radiush) ** 2 < 1:
                    mask[i, j] = 10
                if ((W[i] - centerW) / radiusl1) ** 2 + ((H[j] - centerH) / radiush1) ** 2 < 1:
                    mask[i, j] = 0
                    # fact_rs = 3
        B = np.zeros((fact_rs * width, fact_rs * height))
        Ifft = np.zeros((fact_rs * width, fact_rs * height))

        B = np.matlib.repmat(mask, fact_rs, fact_rs)
        Ifft = np.abs(fftshift(fftn(B)))

        sc = det / len(B);
        J = np.double(scipy.misc.imresize(Ifft, 100 * sc, 'bilinear'))
        J = J ** 2

    img = np.log(0.01 + J)
    img1 = J

    reso = 18
    # Distance between pillar
    pitch = np.double(EZTest.parameters.child('Pitch_2').value())
    r = np.double(EZTest.parameters.child('Radius_2').value())
    shell = np.double(EZTest.parameters.child('Shell_2').value())

    for angl in range(0, 1, 1):
        angle = angl

        width = np.int((pitch * np.cos(np.radians(angle))) / reso)
        height = np.int(pitch / reso)

        # radius of the pillar
        radiush = r / reso
        radiusl = (r * np.cos(np.radians(angle))) / reso

        radiush1 = (r - shell) / reso
        radiusl1 = ((r - shell) * np.cos(np.radians(angle))) / reso

        centerW = width / 2
        centerH = height / 2

        W = np.linspace(1, width, width)
        H = np.linspace(1, height, height)

        mask = np.zeros((width, height))

        for i in range(0, width, 1):
            for j in range(0, height, 1):
                if ((W[i] - centerW) / radiusl) ** 2 + ((H[j] - centerH) / radiush) ** 2 < 1:
                    mask[i, j] = 10
                if ((W[i] - centerW) / radiusl1) ** 2 + ((H[j] - centerH) / radiush1) ** 2 < 1:
                    mask[i, j] = 0
                    # fact_rs = 3
        B = np.zeros((fact_rs * width, fact_rs * height))
        Ifft = np.zeros((fact_rs * width, fact_rs * height))

        B = np.matlib.repmat(mask, fact_rs, fact_rs)
        Ifft = np.abs(fftshift(fftn(B)))

        sc = det / len(B)
        J = np.double(scipy.misc.imresize(Ifft, 100 * sc, 'bilinear'))
        J = J ** 2

    img = np.log(0.01 + J)
    img2 = J

    img = (img2 - img1)
    img = img - np.mean(img)
    EZTest.setImage(img)


def opentest(filepaths):
    import fabio
    for filepath in filepaths:
        img = fabio.open(filepath).data
        img = np.rot90(np.log(img - np.min(img) + 0.01))
        EZTest.setImage(img)
    np.save('C:\\Users\\tpicardjoly\\Desktop\\picture3d.npy', img)


def picture3d():
    points, triangles = loadafmasmesh('C:\\Users\\tpicardjoly\\Desktop\\picture3d.npy')
    np.save('C:\\Users\\tpicardjoly\\Desktop\\picture3d.npy', triangles)


def random3d():
    img = np.random.random((100, 100, 100))
    EZTest.setImage(img)

    np.save('C:\\Users\\tpicardjoly\\Desktop\\random3d.npy', img)
    points, triangles = loadafmasmesh('C:\\Users\\tpicardjoly\\Desktop\\random3d.npy')
    np.save('C:\\Users\\tpicardjoly\\Desktop\\random3d.npy', triangles)


EZTest = base.EZplugin(name='IsvarPlugIn',
                       toolbuttons=[('xicam/gui/icons_28.png', runtest),
                                    ('xicam/gui/3d.png', img3d),
                                    ('xicam/gui/icons_34.png', runtest2),
                                    ('xicam/gui/icons_23.png', run_change),
                                    ('xicam/gui/3d.png', picture3d),
                                    ('xicam/gui/3d.png', random3d)],
                       parameters=[{'name': 'Pitch', 'value': 2200, 'type': 'int'},
                                   {'name': 'Radius', 'value': 600, 'type': 'int'},
                                   {'name': 'Shell', 'value': 98, 'type': 'int'},
                                   {'name': 'Coherence', 'value': 2, 'type': 'int'},
                                   {'name': 'Detector', 'value': 2048, 'type': 'int'},
                                   {'name': 'Pitch_2', 'value': 2200, 'type': 'int'},
                                   {'name': 'Radius_2', 'value': 600, 'type': 'int'},
                                   {'name': 'Shell_2', 'value': 98, 'type': 'int'}],
                       openfileshandler=opentest)
