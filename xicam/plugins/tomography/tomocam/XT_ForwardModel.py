#Fuction to implement the tomographic forward and back-projection kernel in python

import gnufft 
import math
import numpy as np
import afnumpy as afnp 
import afnumpy.fft as af_fft
import scipy.special as sc_spl #For bessel functions
import tomopy
from XT_Common import padmat
#import ipdb

def forward_project(x,params):
    #inputs : x - afnumpy array containing the complex valued image
    #       : params - a list containing all parameters for the NUFFT 

    qxyXrxy = (params['fft2Dshift']*af_fft.fft2(x*params['deapod_filt']*params['fft2Dshift']))/params['Ns'] #real space (rxy) to Fourier space (qxy)
    
    qtXqxy = gnufft.polarsample(params['gxy'],qxyXrxy,params['gkblut'],params['scale'],params['k_r']) #Fourier space to polar coordinates interpolation (qxy to qt)

    rtXqt = params['fftshift1D']((af_fft.ifft(afnp.array(params['fftshift1D_center'](qtXqxy).T))).T)*params['sino_mask'] #Polar cordinates to real space qt to rt 

    return rtXqt 

def back_project(y,params):
    #inputs : y - afnumpy array containing the complex valued array with size of the sinogram 
    #       : params - a list containing all parameters for the NUFFT 

    qtXrt = params['giDq'].reshape((params['Ns'],1))*(params['fftshift1Dinv_center'](af_fft.fft((params['fftshift1D'](y)).T).T)) #Detector space rt to Fourier space qt

    qxyXqt = gnufft.polarsample_transpose(params['gxy'],qtXrt,params['grid'],params['gkblut'],params['scale'],params['k_r'])

    rxyXqxy =params['fft2Dshift']*(af_fft.ifft2(qxyXqt*params['fft2Dshift']))*params['deapod_filt']*params['Ns'] #Fourier to real space : qxy to rxy

    return rxyXqxy


def init_nufft_params(sino,geom):
   # Function to initialize parameters associated with the forward model 
    #inputs : sino - A list contating parameters associated with the sinogram 
    #              Ns : Number of entries in the padded sinogram along the "detector" rows 
    #              Ns_orig :  Number of entries  detector elements per slice
    #              center : Center of rotation in pixels computed from the left end of the detector
    #              angles : An array containg the angles at which the data was acquired in radians
    #       : geom - TBD
    #
     
    KBLUT_LENGTH = 256;
    SCALING_FACTOR = 1.7;#What is this ? 
    k_r=3 #kernel size 2*kr+1
    beta =4*math.pi  
    Ns = sino['Ns']
    Ns_orig = sino['Ns_orig']
    ang = sino['angles']

    q_grid = np.arange(1,sino['Ns']+1) - np.floor((sino['Ns']+1)/2) - 1
    sino['tt'],sino['qq']=np.meshgrid(ang*180/math.pi,q_grid)

    # Preload the Bessel kernel (real components!)
    kblut,KB,KB1D,KB2D=KBlut(k_r,beta,KBLUT_LENGTH) 
    KBnorm=np.array(np.single(np.sum(np.sum(KB2D(np.reshape(np.arange(-k_r,k_r+1),(2*k_r+1,1)),(np.arange(-k_r,k_r+1)))))))
    #print KBnorm
    kblut=kblut/KBnorm*SCALING_FACTOR #scaling fudge factor


    #Normalization (density compensation factor)
#    Dq=KBdensity1(sino['qq'],sino['tt'],KB1,k_r,Ns)';

    # polar to cartesian, centered
    [xi,yi]=pol2cart(sino['qq'],sino['tt']*math.pi/180)
    xi = xi+np.floor((Ns+1)/2)
    yi = yi+np.floor((Ns+1)/2)
   
    params={}
    params['k_r'] = k_r
    params['deapod_filt']=afnp.array(deapodization(Ns,KB,Ns_orig),dtype=afnp.float32)
    params['sino_mask'] = afnp.array(padmat(np.ones((Ns_orig,sino['qq'].shape[1])),np.array((Ns,sino['qq'].shape[1])),0),dtype=afnp.float32)
    params['grid'] = [Ns,Ns] #np.array([Ns,Ns],dtype=np.int32)
    params['scale']= ((KBLUT_LENGTH-1)/k_r)
    params['center'] = afnp.array(sino['center'])
    params['Ns'] = Ns

    # push parameters to gpu and initalize a few in-line functions 
    params['gxi'] = afnp.array(np.single(xi))
    params['gyi'] = afnp.array(np.single(yi))
    params['gxy'] = params['gxi']+1j*params['gyi']
    params['gkblut'] = afnp.array(np.single(kblut))
    params['det_grid'] = np.array(np.reshape(np.arange(0,sino['Ns']),(sino['Ns'],1)))

    #####Generate Ram-Lak/ShepLogan like filter kernel#########
    
    temp_mask=np.ones(Ns)
    kernel=np.ones(Ns)
    if 'filter' in sino:
      temp_r = np.linspace(-1,1,Ns)
      kernel = (Ns)*np.fabs(temp_r)*np.sinc(temp_r/2)
      temp_pos = (1-sino['filter'])/2
      temp_mask[0:np.int16(temp_pos*Ns)]=0
      temp_mask[np.int16((1-temp_pos)*Ns):]=0
    params['giDq']=afnp.array(kernel*temp_mask,dtype=afnp.complex64)
    
    temp = afnp.array((-1)**params['det_grid'],dtype=afnp.float32)
    temp2 = np.array((-1)**params['det_grid'],dtype=afnp.float32)
    temp2 = afnp.array(temp2.reshape(1,sino['Ns']))
    temp3 = afnp.array(afnp.exp(-1j*2*params['center']*(afnp.pi/params['Ns'])*params['det_grid']).astype(afnp.complex64))
    temp4 = afnp.array(afnp.exp(1j*2*params['center']*afnp.pi/params['Ns']*params['det_grid']).astype(afnp.complex64))
    params['fft2Dshift'] = afnp.array(temp*temp2,dtype=afnp.complex64)
    params['fftshift1D'] = lambda x : temp*x
    params['fftshift1D_center'] = lambda x : temp3*x
    params['fftshift1Dinv_center'] = lambda x : temp4*x

################# Back projector params #######################
    xi = xi.astype(np.float32)
    yi = yi.astype(np.float32)
    
#    [s_per_b,b_dim_x,b_dim_y,s_in_bin,b_offset,b_loc,b_points_x,b_points_y] = gnufft.polarbin(xi,yi,params['grid'],4096*4,k_r)
#    params['gs_per_b']=afnp.array(s_per_b,dtype=afnp.int64) #int64
#    params['gs_in_bin']=afnp.array(s_in_bin,dtype=afnp.int64)
#    params['gb_dim_x']= afnp.array(b_dim_x,dtype=afnp.int64)
#    params['gb_dim_y']= afnp.array(b_dim_y,dtype=afnp.int64)
#    params['gb_offset']=afnp.array(b_offset,dtype=afnp.int64)
#    params['gb_loc']=afnp.array(b_loc,dtype=afnp.int64)
#    params['gb_points_x']=afnp.array(b_points_x,dtype=afnp.float32)
#    params['gb_points_y']=afnp.array(b_points_y,dtype=afnp.float32)

    return params

def deapodization(Ns,KB,Ns_orig):

    xx=np.arange(1,Ns+1)-Ns/2-1
    dpz=np.fft.fftshift(np.fft.ifft2(np.fft.fftshift(np.reshape(KB(xx,np.array(0)),(np.size(xx),1))*KB(xx,np.array(0)))))
    # assume oversampling, do not divide outside box in real space:
    msk = padmat(np.ones((Ns_orig,Ns_orig)),np.array((Ns,Ns)),0)
    msk=msk.astype(bool)
    dpz=dpz.real#astype(float)
    dpz[~msk] = 1            #keep the value outside box
    dpz=1/dpz               #deapodization factor truncated
    dpz=dpz/dpz[Ns/2+1,Ns/2+1] #scaling
    return dpz

    
def KBlut(k_r,beta,nlut):
    kk=np.linspace(0,k_r,nlut)
    kblut = KB2( kk, 2*k_r, beta)
    scale = (nlut-1)/k_r
    kbcrop = lambda x: (np.abs(x)<=k_r)
    KBI = lambda x: np.int16(np.abs(x)*scale-np.floor(np.abs(x)*scale))
    KB1D = lambda x: (np.reshape(kblut[np.int16(np.floor(np.abs(x)*scale)*kbcrop(x))],x.shape)*KBI(x)+np.reshape(kblut[np.int16(np.ceil(np.abs(x)*scale)*kbcrop(x))],x.shape)*(1-KBI(x)))*kbcrop(x)
    KB=lambda x,y: KB1D(x)*KB1D(y)
    KB2D=lambda x,y: KB1D(x)*KB1D(y)
    return kblut, KB, KB1D,KB2D

def KB2(x, k_r, beta):
    w = sc_spl.iv(0, beta*np.sqrt(1-(2*x/k_r)**2)) 
    w=w/np.abs(sc_spl.iv(0, beta))
    w=(w*(x<=k_r))
    return w

def cart2pol(x, y):
    rho = np.sqrt(x**2 + y**2)
    phi = np.arctan2(y, x)
    return(rho, phi)

def pol2cart(rho, phi):
    x = rho * np.cos(phi)
    y = rho * np.sin(phi)
    return(x, y)

#def densityCompensation(qq,tt,KB,nj,Ns):
#    nb=100; #TODO : Why 100 ? Venkat
#    [nt,nq]=size(qq);   
    #crop repeated angles
#    ii=(tt(:,1))-min(tt(:,1))<180;
#    qq1=qq(ii,:);
#    tt1=tt(ii,:);
#    qq1=-fliplr(qq1);
#    [xi,yi]=pol2cart(tt1*pi/180,qq1);
