import numpy as np
import tomopy
import afnumpy as afnp

def padmat(x,siz,value):
# function y=padmat(x,size,vals)
# pads x to size with constant values (vals), 
# centers the matrix in the middle.
    n=siz[0]
    if siz.size < 2:
      m=n
    elif siz.size == 2:
      m=siz[1]
    else:
      (n,m) = siz.shape
    
    [N,M]=x.shape

    y=np.zeros((n,m))+value
    y[0:N,0:M]=x
    y=np.roll(np.roll(y,np.int16(np.fix((n-N)/2)),axis=0),np.int16(np.fix((m-M)/2)),axis=1)
    return y


def padmat_v2(x,siz,value,y):
# function y=padmat(x,size,vals)
# pads x to size with constant values (vals), 
# centers the matrix in the middle.
    n=siz[0]
    if siz.size < 2:
      m=n
    elif siz.size == 2:
      m=siz[1]
    else:
      (n,m) = siz.shape
    
    [N,M]=x.shape
    y[0:N,0:M]=x
    y=np.roll(np.roll(y,afnp.int16(np.fix((n-N)/2)),axis=0),np.int16(afnp.fix((m-M)/2)),axis=1)
    return y
