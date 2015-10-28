import fabio
import numpy
import pyfits
import os

acceptableexts = '.fits .edf .tif'


def loadsinglepath(path):
    """
    :type path : str
    :param path:
    :return:
    """

    if os.path.splitext(path)[1] in '.fits .edf .tif':
        if path.split('.')[-1] == 'fits':
            data = pyfits.open(path)[2].data
        else:
            data = fabio.open(path).data

        txtpath = '.'.join(path.split('.')[:-1]) + '.txt'
        # print txtpath
        if os.path.isfile(txtpath):
            with open(txtpath, 'r') as f:
                lines = f.readlines()
                paras = dict()
                i = 0
                for line in lines:
                    cells = line.split(':')
                    if cells.__len__() == 2:
                        paras[cells[0]] = float(cells[1])
                    elif cells.__len__() == 1:
                        i += 1
                        paras['Unknown' + str(i)] = str(cells[0])
            # print paras
            return data, paras
        else:
            return data, None

    return None, None

    # except TypeError:
    #   print('Failed to load',path,', its probably not an image format I understand.')
    #  return None,None


def loadstichted(filepath2, filepath1):
    (data1, paras1) = loadsinglepath(filepath1)
    (data2, paras2) = loadsinglepath(filepath2)

    positionY1=paras1['Detector Vertical']
    positionY2=paras2['Detector Vertical']
    positionX1=paras1['Detector Horizontal']
    positionX2=paras2['Detector Horizontal']
    deltaX= round((positionX2-positionX1)/0.172)
    deltaY= round((positionY2-positionY1)/0.172)
    padtop2=0
    padbottom1=0
    padtop1=0
    padbottom2=0
    padleft2=0
    padright1=0
    padleft1=0
    padright2=0

    if deltaY<0:
        padtop2=abs(deltaY)
        padbottom1=abs(deltaY)
    else:
        padtop1=abs(deltaY)
        padbottom2=abs(deltaY)

    if deltaX<0:
        padleft2=abs(deltaX)
        padright1=abs(deltaX)

    else:
        padleft1=abs(deltaX)
        padright2=abs(deltaX)

    d2=numpy.pad(data2,((padtop2,padbottom2),(padleft2,padright2)),'constant')
    d1=numpy.pad(data1,((padtop1,padbottom1),(padleft1,padright1)),'constant')

    mask2=numpy.pad((data2>0),((padtop2,padbottom2),(padleft2,padright2)),'constant')
    mask1=numpy.pad((data1>0),((padtop1,padbottom1),(padleft1,padright1)),'constant')
    with numpy.errstate(divide='ignore'):
        data=(d1+d2)/(mask2+mask1)
    return data


def loadsecondfile(filepath1):
    if '_lo_' in filepath1:
        # print "input: lo / output: hi"
        filepath2 = filepath1.replace('_lo_', '_hi_')

    elif '_hi_' in filepath1:
        # print "input: hi / output: lo"
        filepath2 = filepath1.replace('_hi_', '_lo_')

    else:
        print "file don't contain hi or lo, just 1 file"
        return None

    return filepath1, filepath2


def loadpath(path):
    paths = loadsecondfile(path)
    paras = None
    if paths is not None:
        data = loadstichted(*paths)
    else:
        data, paras = loadsinglepath(path)
    return data, paras


#
# if __name__=='__main__':
# from PIL import Image
# data,paras=loadpath('/home/remi/PycharmProjects/saxs-on/samples/AgB_5s_hi_2m.edf')
# print data
#     IM=Image.fromarray(data)
#     IM.show()









    ##################################################################"
    #
    # tailleIM=(taille[0]-2*deltaY,taille[1])
    # IM=zeros((tailleIM))
    #
    # for i in range(0,np.size(IM,0)):
    #     for j in range(0,np.size(IM,1)):
    #         if data1[i,j]==0:
    #             IM[i,j]=data2[i+deltaY,j]
    #         elif data2[i+deltaY,j]==0:
    #             IM[i,j]=data1[i,j]
    #         else:
    #             IM[i,j]=(data1[i,j]+data2[i+deltaY,j])/2
    # Ima=Image.fromarray(IM)
# Ima.show()




