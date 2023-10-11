import numpy as np
import scipy
from scipy.io import loadmat
import cv2 as cv
import math
import copy
from skimage.util import random_noise
import math as m

inputsFolder = 'inputs/'

# TODO erase the use of crops, in the drawing of a single fish
crop_b = np.array([0, 487, 0, 647])
crop_s1 = np.array([0, 487, 0, 647])
crop_s2 = np.array([0, 487, 0, 647])
# Old crops, these are wrong
# crop_b = np.array([-12.5, 474.5, -6.5, 640.5])
# crop_s1 = np.array([-26.5, 460.5, 12.5, 659.5])
# crop_s2 = np.array([-22.5, 464.5, -8.5, 638.5])


def uint8(a):
    """
       Function that turns values into uint8 values like matlab

       Args:
           a (numpy array or float): value or values you with to transform

       Returns:
            the value in uint8 form
    """
    a = roundHalfUp(a)

    if np.ndim(a) == 0:
        if a <0:
            a = 0
        if a > 255:
            a = 255
    else:
        a[a>255]=255
        a[a<0]=0
    return a


def normpdf(x, mean, sd):
    """
       Function that gets the value from a normal distribution

       Args:
           x (numpy array): x value
           mean (float): mean value for the distribution
           sd (float): standard deviation value for the distribution

       Returns:
            the value from the normal distribution at that particular x value
    """
    var = float(sd)**2
    denom = (2*np.pi*var)**.5
    num = np.exp(-(x-float(mean))**2/(2*var))
    return num/denom

def bellymodel(x, y, z, seglen, brightness, size_lut):
    """
       Function that returns a numpy array of the belly model

       Args:
           x (numpy array): x coordinates
           y (numpy array): y coordinates
           z (numpy array): z coordinates
           seglen (float): segment length
           brightness (float): brightness of the belly
           size_lut (float): size of the box in which the fish is rendered

       Returns:
           look up table for the belly model
    """
    belly_w = seglen * (0.499 + (np.random.rand() - 0.5)*0.03)
    belly_l = seglen * (1.2500 + (np.random.rand() - 0.5)*0.07)

    belly_h = seglen * (0.7231 + (np.random.rand() - 0.5)*0.03)
    c_belly = 1.0541 + (np.random.rand() - 0.5)*0.03

    pt_original = np.empty((3,3))
    pt_original[:,1] = [size_lut/2, size_lut/2, size_lut/2]
    pt_original[:,0] = pt_original[:,1] - [seglen, 0, 0]
    pt_original[:,2] = pt_original[:,1] + [seglen, 0, 0]
    belly_c = [c_belly*pt_original[0,1] + (1-c_belly)*pt_original[0,2], c_belly*pt_original[1,1] + (1-c_belly)*pt_original[1,2], pt_original[2,1] - seglen/(6+(np.random.rand() - 0.5)*0.05)]


    XX = x - belly_c[0]
    YY = y - belly_c[1]
    ZZ = z - belly_c[2]

    belly_model = np.exp(-2*(XX * XX / (2 * belly_l**2) + YY*YY/(2* belly_w**2) + ZZ*ZZ/(2* belly_h**2) - 1))
    belly_model = belly_model*brightness
    return belly_model

def calc_proj_w_refra_cpu(coor_3d,proj_params = inputsFolder + 'proj_params_101019_corrected_new.mat' ):
    """
        Function calculates the projections of a set of points

        Args:
            coor_3d (numpy array): array in which the first row are the x coordinates
                second row are the y coordinates, third rows are the z coordinates
            proj_params (string): path to projection parameters

       Returns:
            coor_b (numpy array): array in which the first row are x coordinates and
                second row are the y coordinates for the points in the bottom view
            coor_s1 (numpy array): array in which the first row are x coordinates and
                second row are the y coordinates for the points in side view 1
            coor_s2 (numpy array): array in which the first row are x coordinates and
                second row are the y coordinates for the points in side view 2
    """

    mat = loadmat(proj_params)
    proj_params_mat = mat['proj_params']

    fa1p00 = proj_params_mat[0,0]
    fa1p10 = proj_params_mat[0,1]
    fa1p01 = proj_params_mat[0,2]
    fa1p20 = proj_params_mat[0,3]
    fa1p11 = proj_params_mat[0,4]
    fa1p30 = proj_params_mat[0,5]
    fa1p21 = proj_params_mat[0,6]
    fa2p00 = proj_params_mat[1,0]
    fa2p10 = proj_params_mat[1,1]
    fa2p01 = proj_params_mat[1,2]
    fa2p20 = proj_params_mat[1,3]
    fa2p11 = proj_params_mat[1,4]
    fa2p30 = proj_params_mat[1,5]
    fa2p21 = proj_params_mat[1,6]
    fb1p00 = proj_params_mat[2,0]
    fb1p10 = proj_params_mat[2,1]
    fb1p01 = proj_params_mat[2,2]
    fb1p20 = proj_params_mat[2,3]
    fb1p11 = proj_params_mat[2,4]
    fb1p30 = proj_params_mat[2,5]
    fb1p21 = proj_params_mat[2,6]
    fb2p00 = proj_params_mat[3,0]
    fb2p10 = proj_params_mat[3,1]
    fb2p01 = proj_params_mat[3,2]
    fb2p20 = proj_params_mat[3,3]
    fb2p11 = proj_params_mat[3,4]
    fb2p30 = proj_params_mat[3,5]
    fb2p21 = proj_params_mat[3,6]
    fc1p00 = proj_params_mat[4,0]
    fc1p10 = proj_params_mat[4,1]
    fc1p01 = proj_params_mat[4,2]
    fc1p20 = proj_params_mat[4,3]
    fc1p11 = proj_params_mat[4,4]
    fc1p30 = proj_params_mat[4,5]
    fc1p21 = proj_params_mat[4,6]
    fc2p00 = proj_params_mat[5,0]
    fc2p10 = proj_params_mat[5,1]
    fc2p01 = proj_params_mat[5,2]
    fc2p20 = proj_params_mat[5,3]
    fc2p11 = proj_params_mat[5,4]
    fc2p30 = proj_params_mat[5,5]
    fc2p21 = proj_params_mat[5,6]

    npts = coor_3d.shape[1]

    coor_b = np.zeros((2,npts))
    coor_s1 = np.zeros((2, npts))
    coor_s2 = np.zeros((2, npts))

    coor_b[0,:] = fa1p00 + fa1p10*coor_3d[2,:] + fa1p01*coor_3d[0,:] + fa1p20*(coor_3d[2,:]**2) + fa1p11*coor_3d[2,:]*coor_3d[0,:] + fa1p30*(coor_3d[2,:]**3) + fa1p21*(coor_3d[2,:]**2)*coor_3d[0,:]
    coor_b[1,:] = fa2p00 + fa2p10*coor_3d[2,:] + fa2p01*coor_3d[1,:] + fa2p20*(coor_3d[2,:]**2) + fa2p11*coor_3d[2,:]*coor_3d[1,:] + fa2p30*(coor_3d[2,:]**3) + fa2p21*(coor_3d[2,:]**2)*coor_3d[1,:]
    coor_s1[0,:] = fb1p00 + fb1p10*coor_3d[0,:] + fb1p01*coor_3d[1,:] + fb1p20*(coor_3d[0,:]**2) + fb1p11*coor_3d[0,:]*coor_3d[1,:] + fb1p30*(coor_3d[0,:]**3) + fb1p21*(coor_3d[0,:]**2)*coor_3d[1,:]
    coor_s1[1,:] = fb2p00 + fb2p10*coor_3d[0,:] + fb2p01*coor_3d[2,:] + fb2p20*(coor_3d[0,:]**2) + fb2p11*coor_3d[0,:]*coor_3d[2,:] + fb2p30*(coor_3d[0,:]**3) + fb2p21*(coor_3d[0,:]**2)*coor_3d[2,:]
    coor_s2[0,:] = fc1p00 + fc1p10*coor_3d[1,:] + fc1p01*coor_3d[0,:] + fc1p20*(coor_3d[1,:]**2) + fc1p11*coor_3d[1,:]*coor_3d[0,:] + fc1p30*(coor_3d[1,:]**3) + fc1p21*(coor_3d[1,:]**2)*coor_3d[0,:]
    coor_s2[1,:] = fc2p00 + fc2p10*coor_3d[1,:] + fc2p01*coor_3d[2,:] + fc2p20*(coor_3d[1,:]**2) + fc2p11*coor_3d[1,:]*coor_3d[2,:] + fc2p30*(coor_3d[1,:]**3) + fc2p21*(coor_3d[1,:]**2)*coor_3d[2,:]

    return coor_b, coor_s1, coor_s2

def eye1model(x,y,z,seglen,brightness,size_lut,rnd):
    """
        Function that creates the model of the first eye

        Args:
            x (numpy array): x coordinates
            y (numpy array): y coordinates
            z (numpy array): z coordinates
            seglen (float): segment length
            brightness (float): how bright the fish should be
            size_lut (float): size of the box in which the fish is rendered
            rnd (numpy array): array of size 4 containing random values for eye

       Returns:
            eye1_model (numpy array): second eye model
            eye1_c (numpy array): the second eyes coordinates
    """
    d_eye = seglen * (0.8556 + (rnd[0] - 0.5) * 0.05) # 1.4332 # 0.83
    c_eyes = 1.4230 # 1.3015
    eye1_w = seglen * (0.3197 + (rnd[1] - 0.5) * 0.02)
    eye1_l = seglen * (0.4756 + (rnd[2] - 0.5) * 0.02)
    eye1_h = seglen * (0.2996 + (rnd[3] - 0.5) * 0.02)

    pt_original = np.empty([3,3])
    pt_original[:, 1] = np.array([size_lut / 2, size_lut / 2, size_lut / 2])
    pt_original[:, 0] = np.array(pt_original[:, 1] - [seglen, 0, 0])
    pt_original[:, 2] = np.array(pt_original[:, 1] + [seglen, 0, 0])

    eye1_c = [c_eyes * pt_original[0, 0] + (1 - c_eyes) * pt_original[0, 1], c_eyes * pt_original[1, 0] + (1 - c_eyes) * pt_original[1, 1] + d_eye / 2, pt_original[2, 1] - seglen / 7.3049]


    XX = x - eye1_c[0]
    YY = y - eye1_c[1]
    ZZ = z - eye1_c[2]

    eye1_model = np.exp(-1.2 * (XX * XX / (2 * eye1_l**2) + YY * YY / (2 * eye1_w**2) + ZZ * ZZ / (2 * eye1_h**2) - 1))
    eye1_model = eye1_model * brightness

    return eye1_model, eye1_c

def eye2model(x, y, z, seglen, brightness, size_lut, rnd):
    """
        Function that creates the model of the second eye

        Args:
            x (numpy array): x coordinates
            y (numpy array): y coordinates
            z (numpy array): z coordinates
            seglen (float): segment length
            size_lut (float): size of the box in which the fish is rendered

       Returns:
            eye2_model (numpy array): second eye model
            eye2_c (numpy array): the second eyes coordinates
    """

    d_eye = seglen * (0.8556 + (rnd[0] - 0.5)*0.05) # 1.4332# 0.83
    c_eyes = 1.4230 #1.3015

    eye2_w = seglen * (0.3197 + (rnd[1] - 0.5)*0.02)
    eye2_l = seglen * (0.4756 + (rnd[2] - 0.5)*0.02)
    eye2_h = seglen * (0.2996 + (rnd[3] - 0.5)*0.02)

    pt_original = np.empty((3,3))
    pt_original[:,1] = [size_lut/2, size_lut/2, size_lut/2]
    pt_original[:,0] = pt_original[:,1] - [seglen, 0, 0]
    pt_original[:,2] = pt_original[:,1] + [seglen, 0, 0]
    eye2_c = [c_eyes*pt_original[0,0] + (1-c_eyes)*pt_original[0,1], c_eyes*pt_original[1,0] + (1-c_eyes)*pt_original[1,1] - d_eye/2, pt_original[2,1] - seglen/7.3049]

    XX = x - eye2_c[0]
    YY = y - eye2_c[1]
    ZZ = z - eye2_c[2]

    eye2_model = np.exp(-1.2*(XX*XX/(2*eye2_l**2) + YY*YY/(2*eye2_w**2) + ZZ*ZZ/(2*eye2_h**2) - 1))
    eye2_model = eye2_model*brightness

    return eye2_model, eye2_c

def roundHalfUp(a):
    """
       Function to round the way motlab would

       Args:
           a (numpy array or float)

       Returns:
            a rounded the way that matlab would
       """
    return (np.floor(a) + np.round(a - (np.floor(a) - 1)) - 1)

def gen_lut_s_tail(n, seglenidx, d1, d2, a):
    """
       Function to generate look up table for side view

       Args:
           n (numpy array): idx for ball size and thickness
           nseglen (float): seglen
           d1 (int): distance of the box in the x direction
           d2 (int): distance of the box in the y direction
           a (float): angle
           rand (float): random number used to affect the ball size and thickness

       Returns:
            lut (numpy array): numpy array representing tail from the side view
    """

    size_lut = 15

    size_half = (size_lut + 1) / 2

    imblank = np.zeros((size_lut, size_lut))

    imageSizeX = size_lut

    imageSizeY = size_lut

    random_number = np.random.normal(1.1,0.1)

    # size of the balls in the model

    temp = [2.5, 2.4, 2.3, 2.2, 1.8, 1.5, 1.3, 1.2]
    temp = np.array(temp)
    ballsize = random_number * temp

    # thickness of the sticks in the model
    temp = [8, 7, 6, 5, 4, 3, 2.5, 2.5]
    temp = np.array(temp)
    thickness = random_number * temp

    # brightness of the tail

    b_tail = [0.5, 0.45, 0.4, 0.32, 0.28, 0.24, 0.22, 0.20]
    b_tail = np.array(b_tail)

    x = np.linspace(1, imageSizeX, imageSizeX)
    y = np.linspace(1, imageSizeY, imageSizeY)

    [columnsInImage0, rowsInImage0] = np.meshgrid(x, y)

    radius = ballsize[n]

    th = thickness[n]

    # p_max = scipy.stats.norm.pdf(0, loc=0, scale=th)
    p_max = normpdf(0,0,th)

    bt_gradient = b_tail[n] / b_tail[n - 1]

    seglen = 0.2 * seglenidx

    bt = b_tail[n - 1] * (1 - 0.02 * seglenidx)

    centerX = size_half + d1 / 5
    centerY = size_half + d2 / 5

    columnsInImage = columnsInImage0
    rowsInImage = rowsInImage0

    ballpix = (rowsInImage - centerY) ** 2 + (columnsInImage - centerX) ** 2 <= radius ** 2

    ballpix = uint8(uint8(uint8(uint8(ballpix) * 255) * bt) * 0.85)

    t = 2 * np.pi * (a - 1) / 180

    pt = np.zeros((2, 2))

    R = [[np.cos(t), -np.sin(t)], [np.sin(t), np.cos(t)]]

    vec = np.matmul(R, np.array([[seglen], [0]]))

    pt[:, 0] = np.array([size_half + d1 / 5, size_half + d2 / 5])

    pt[:, 1] = pt[:, 0] + vec[:, 0]
    stickpix = imblank

    columnsInImage = columnsInImage0
    rowsInImage = rowsInImage0

    if (pt[0, 1] - pt[0, 0]) != 0:

        slope = (pt[1, 1] - pt[1, 0]) / (pt[0, 1] - pt[0, 0])

        # vectors perpendicular to the line segment

        # th is the thickness of the sticks in the model

        vp = np.array([[-slope], [1]]) / np.linalg.norm(np.array([[-slope], [1]]))

        # one vertex of the rectangle

        # POSSIBLE SOURCE OF ERROR
        V1 = pt[:, 1] - vp[:, 0] * th

        # two sides of the rectangle

        s1 = 2 * vp * th

        s2 = pt[:, 0] - pt[:, 1]

        # find the pixels inside the rectangle

        r1 = rowsInImage - V1[1]

        c1 = columnsInImage - V1[0]

        # inner products

        ip1 = r1 * s1[1] + c1 * s1[0]

        ip2 = r1 * s2[1] + c1 * s2[0]

        stickpix_bw = (ip1 > 0) * (ip1 < np.dot(s1[:, 0], s1[:, 0])) * (ip2 > 0) * (ip2 < np.dot(s2, s2))

    else:
        stickpix_bw = (rowsInImage < max(pt[1, 1], pt[1, 0])) * (rowsInImage > min(pt[1, 1], pt[1, 0])) * (
                    columnsInImage < pt[0, 1] + th) * (columnsInImage > pt[0, 1] - th)

    # the brightness of the points on the stick is a function of its

    # distance to the segment

    #[ys, xs] = ind2sub(stickpix_bw)
    idx_bw = np.argwhere(stickpix_bw >0)
    ys = idx_bw[:, 0]
    xs = idx_bw[:, 1]

    px = pt[0, 1] - pt[0, 0]

    py = pt[1, 1] - pt[1, 0]

    pp = px * px + py * py

    # the distance between a pixel and the fish backbone

    d_radial = np.zeros((max(ys.shape), 1))

    # the distance between a pixel and the anterior end of the

    # segment (0 < d_axial < 1)

    b_axial = np.zeros((max(ys.shape), 1))

    for i in range(0, max(ys.shape)):
        u = (((xs[i] + 1) - pt[0, 0]) * px + ((ys[i] + 1) - pt[1, 0]) * py) / pp

        dx = pt[0, 0] + u * px - xs[i] - 1

        dy = pt[1, 0] + u * py - ys[i] - 1

        d_radial[i] = dx * dx + dy * dy

        b_axial[i] = 1 - (1 - bt_gradient) * u * 0.9

    # b_stick = scipy.stats.norm.pdf(d_radial, 0, th) / p_max * 255
    b_stick = normpdf(d_radial, 0, th) / p_max * 255

    b_stick = uint8(b_stick)

    for i in range(0, max(ys.shape)):
        stickpix[ys[i], xs[i]] = uint8(b_stick[i] * b_axial[i])

    stickpix = stickpix * bt
    stickpix = uint8(stickpix)

    graymodel = np.maximum(ballpix, stickpix)
    graymodel = uint8(graymodel)

    return graymodel


def gen_lut_b_tail(n, nseglen, d1, d2, a):
    """
       Function to generate look up table for bottom view

       Args:
           n (numpy array): idx for ball size and thickness
           nseglen (float): seglen
           d1 (int): distance of the box in the x direction
           d2 (int): distance of the box in the y direction
           a (float): angle
           rand (float): random number used to affect the ball size and thickness

       Returns:
            lut (numpy array): numpy array representing tail from the bottom view
    """
    size_lut = 19
    size_half = (size_lut+1)/2


    random_number = np.random.normal(0.9,0.1)
    ballsize = random_number * np.array([3,2,2,2,2,1.5,1.2,1.2,1])
    thickness = random_number*np.array([7,6,5.5,5,4.5,4,3.5,3])
    b_tail = [0.7,0.55,0.45,0.40,0.32,0.28,0.2,0.15]


    imageSizeX = size_lut
    imageSizeY = size_lut
    x = np.linspace(1,imageSizeX,imageSizeX)
    y = np.linspace(1,imageSizeY,imageSizeY)
    [columnsInImage0, rowsInImage0] = np.meshgrid(x, y)


    imblank = np.zeros((size_lut,size_lut))


    radius = ballsize[n]
    th = thickness[n]
    bt = b_tail[n-1]
    bt_gradient = b_tail[n]/b_tail[n-1]

    # p_max = scipy.stats.norm.pdf(0,loc= 0,scale= th)
    p_max = normpdf(0,0,th)

    seglen = 5 + 0.2 * nseglen
    centerX = size_half + d1/5
    centerY = size_half + d2/5

    columnsInImage = columnsInImage0
    rowsInImage = rowsInImage0

    ballpix = (rowsInImage - centerY)**2 + (columnsInImage - centerX)**2 <= radius**2

    # Following lines are equivalent to:
    # ballpix = uint8(ballpix); ballpix = ballpix * 255 * bt * .85
    ballpix = uint8(ballpix)

    ballpix = ballpix * 255
    ballpix = uint8(ballpix)

    ballpix = ballpix * bt
    ballpix = uint8(ballpix)

    ballpix = ballpix * 0.85
    ballpix = uint8(ballpix)


    t = 2*np.pi*(a-1)/360

    pt = np.zeros((2,2))

    R = [[np.cos(t),-np.sin(t)],[np.sin(t),np.cos(t)]]

    vec = np.matmul(R,np.array([[seglen] ,[0]]))

    pt[:,0] = np.array([size_half + d1/5, size_half + d2/5])
    pt[:,1] = pt[:,0] + vec[:,0]

    stickpix = imblank

    columnsInImage = columnsInImage0

    rowsInImage = rowsInImage0

    if (pt[0,1] - pt[0,0]) != 0:

        slope = (pt[1,1] - pt[1,0])/(pt[0,1] - pt[0,0])

        #vectors perpendicular to the line segment

        #th is the thickness of the sticks in the model

        vp = np.array([[-slope],[1]]) / np.linalg.norm(np.array([[-slope],[1]]))

        # one vertex of the rectangle

        V1 = pt[:,1] - vp[:,0] * th


        #two sides of the rectangle

        s1 = 2 * vp * th

        s2 = pt[:,0] - pt[:,1]

        # find the pixels inside the rectangle

        r1 = rowsInImage - V1[1]

        c1 = columnsInImage - V1[0]

        #inner products

        ip1 = r1 * s1[1] + c1 * s1[0]

        ip2 = r1 * s2[1] + c1 * s2[0]


        stickpix_bw = (ip1 > 0) * (ip1 < np.dot(s1[:,0], s1[:,0])) * (ip2 > 0) * (ip2 < np.dot(s2,s2))

    else:
        stickpix_bw = (rowsInImage < max(pt[1,1],pt[1,0])) * (rowsInImage > min(pt[1,1],pt[1,0])) * (columnsInImage < pt[0,1] + th) * (columnsInImage > pt[0,1] - th)

    # the brightness of the points on the stick is a function of its

    # distance to the segment

    #[ys,xs] = ind2sub(stickpix_bw)
    idx_bw = np.argwhere(stickpix_bw >0)
    ys = idx_bw[:, 0]
    xs = idx_bw[:, 1]

    px = pt[0,1] - pt[0,0]

    py = pt[1,1] - pt[1,0]

    pp = px*px + py*py

    # the distance between a pixel and the fish backbone

    ysLength = max(ys.shape)

    d_radial = np.zeros((max(ys.shape),1))

    # the distance between a pixel and the anterior end of the

    # segment (0 < d_axial < 1)

    b_axial = np.zeros((max(ys.shape),1))


    for i in range(0,max(ys.shape)):

        u = (((xs[i]+1) - pt[0,0]) * px + ((ys[i]+1) - pt[1,0]) * py) / pp

        dx = pt[0,0] + u * px - xs[i]-1

        dy = pt[1,0] + u * py - ys[i]-1

        d_radial[i] = dx*dx + dy*dy

        b_axial[i] = 1 - (1 - bt_gradient) * u * 0.9

    # b_stick = scipy.stats.norm.pdf(d_radial, 0, th)/p_max * 255
    b_stick = normpdf(d_radial, 0, th)/p_max * 255

    b_stick = uint8(b_stick)

    for i in range(0,max(ys.shape)):
        stickpix[ys[i],xs[i]] = uint8(b_stick[i]*b_axial[i])

    stickpix = stickpix * bt
    stickpix = uint8(stickpix)

    graymodel = np.maximum(ballpix,stickpix)
    return uint8(graymodel)

def headmodel(x, y, z, seglen, brightness, size_lut):
    """
       Function to return numpy array representing head model

       Args:
           x (numpy array): x coordinates
           y (numpy array): y coordinates
           z (numpy array): z coordinates
           seglen (float): segment length
           indices (numpy array): valid indices for the coordinates
           size_lut (int): size in of the box in which the fish is rendered

       Returns:
            headmodel (numpy array): represents headmodel
       """
    head_w = seglen * (0.6962 + (np.random.rand() - 0.5)*0.03)
    head_l = seglen * (0.8475 + (np.random.rand() - 0.5)*0.03)
    head_h = seglen * (0.7926 + (np.random.rand() - 0.5)*0.03)

    c_head = 1.1771

    pt_original = np.empty((3,3))
    pt_original[:,1] = [size_lut/2, size_lut/2 , size_lut/2]
    pt_original[:,0] = pt_original[:,1] - [seglen, 0, 0]
    pt_original[:,2] = pt_original[:,1] + [seglen, 0, 0]

    head_c = [c_head*pt_original[0,0] + (1-c_head)*pt_original[0,1], c_head*pt_original[1,0] + (1-c_head)*pt_original[1,1], pt_original[2,1] - seglen/(9.3590 + (np.random.rand() - 0.5)*0.05)]

    XX = x - head_c[0]
    YY = y - head_c[1]
    ZZ = z - head_c[2]

    head_model = np.exp(-2*(XX*XX/(2*head_l**2) + YY*YY/(2*head_w**2) + ZZ*ZZ/(2*head_h**2) - 1))
    head_model = head_model*brightness

    return head_model


def project_camera_copy(model, X, Y, Z, proj_params, indices, cb, cs1, cs2):
    """
       Function to project the model according to the camera views

       Args:
           model (numpy array): array representing model
           x (numpy array): x coordinates
           y (numpy array): y coordinates
           z (numpy array): z coordinates
           proj_params (numpy array): path to projection parameters
           indices (numpy array): valid indices for the coordinates
           cb (numpy array): coordinates in bottom view
           cs1 (numpy array): coordinates in side view 1
           cs2 (numpy array): coordinates in side view 2

       Returns:
            projection_b (numpy array): projection of the model in the bottom view
            projection_s1 (numpy array): projection of the model in view s1
            projection_s2 (numpy array): projection of the model in view s2
    """

    (coor_b, coor_s1, coor_s2) = calc_proj_w_refra_cpu(np.array([X, Y, Z]), proj_params)

    coor_b[0,:] = coor_b[0,:] - cb[2]
    coor_b[1,:] = coor_b[1,:] - cb[0]
    coor_s1[0,:] = coor_s1[0,:] - cs1[2]
    coor_s1[1,:] = coor_s1[1,:] - cs1[0]
    coor_s2[0,:] = coor_s2[0,:] - cs2[2]
    coor_s2[1,:] = coor_s2[1,:] - cs2[0]

    projection_b = np.zeros((int(cb[1] - cb[0] + 1) , int(cb[3] - cb[2] + 1)))
    projection_s1 = np.zeros((int(cs1[1] - cs1[0] +1),int( cs1[3] - cs1[2] +1)))
    projection_s2 = np.zeros((int(cs2[1] - cs2[0] +1),int( cs2[3] - cs2[2] +1)))

    sz_b = np.shape(projection_b)
    sz_s1 = np.shape(projection_s1)
    sz_s2 = np.shape(projection_s2)

    count_mat_b = np.zeros(np.shape(projection_b)) + 0.0001
    count_mat_s1 = np.zeros(np.shape(projection_s1)) + 0.0001
    count_mat_s2 = np.zeros(np.shape(projection_s2)) + 0.0001



    length = max(indices.shape)

    x = np.linspace(0,length-1,length)
    x = x.astype(int)


    fval = np.logical_or(np.floor(coor_b[1,x]) > sz_b[0]-1, np.floor(coor_b[0,x]) > sz_b[1]-1)
    sval = np.logical_or(np.floor(coor_b[1,x]) < 0,np.floor(coor_b[0,x] ) < 0)

    finval = np.logical_not(np.logical_or(fval,sval))
    model = np.array(model)

    index1 = (np.floor(coor_b[1, x[finval]])).astype(int)
    index2 = (np.floor(coor_b[0, x[finval]])).astype(int)

    values = model[(indices[x[finval]]).astype(int)]

    np.add.at(projection_b, (index1, index2), values)
    np.add.at(count_mat_b, (index1, index2), 1)

    #projection_b = projection_b / count_mat_b
    projection_b = np.divide(projection_b,count_mat_b)

    i = np.linspace(0,length-1,length)
    i = i.astype(int)

    fval = np.logical_or(np.floor(coor_s1[1, i]) > sz_s1[0]-1,np.floor(coor_s1[0, i]) > sz_s1[1]-1)
    sval = np.logical_or(np.floor(coor_s1[1, i]) < 0,np.floor(coor_s1[0, i]) < 0)
    finval = np.logical_not(np.logical_or(fval,sval))

    index1 = (np.floor(coor_s1[1, i[finval]])).astype(int)
    index2 = (np.floor(coor_s1[0, i[finval]])).astype(int)

    values = model[(indices[i[finval]]).astype(int)]

    np.add.at(projection_s1,(index1,index2),values)
    np.add.at(count_mat_s1,(index1,index2),1)

    #projection_s1 = projection_s1 / count_mat_s1
    projection_s1 = np.divide(projection_s1,count_mat_s1)

    x = np.linspace(0, length - 1, length)
    x = x.astype(int)

    fval = np.logical_or(np.floor(coor_s2[1, x]) > sz_s2[0] - 1, np.floor(coor_s2[0, x]) > sz_s2[1] - 1)
    sval = np.logical_or(np.floor(coor_s2[1, x]) < 0, np.floor(coor_s2[0, x]) < 0)

    finval = np.logical_not(np.logical_or(fval, sval))

    index1 = (np.floor(coor_s2[1, x[finval]])).astype(int)
    index2 = (np.floor(coor_s2[0, x[finval]])).astype(int)

    values = model[(indices[x[finval]]).astype(int)]

    np.add.at(projection_s2, (index1, index2), values)
    np.add.at(count_mat_s2, (index1, index2), 1)

    #projection_s2 = projection_s2 / count_mat_s2
    projection_s2 = np.divide(projection_s2,count_mat_s2)

    return projection_b,projection_s1,projection_s2

def Rx(theta):
    return np.matrix([[1, 0, 0],
                      [0, m.cos(theta), -m.sin(theta)],
                      [0, m.sin(theta), m.cos(theta)]])


def Ry(theta):
    return np.matrix([[m.cos(theta), 0, m.sin(theta)],
                      [0, 1, 0],
                      [-m.sin(theta), 0, m.cos(theta)]])

def Rz(theta):
    return np.matrix([[m.cos(theta), -m.sin(theta), 0],
                      [m.sin(theta), m.cos(theta), 0],
                      [0, 0, 1]])

def reorient_model(model, x_c, y_c, z_c, heading, inclination, roll, ref_vec, hinge):
    """
       Function used rotate points

       Args:
           model (numpy array): array representing model
           x_c (numpy array): x coordinates
           y_c (numpy array): y coordinates
           z_c (numpy array): z coordinates
           heading (float): rotation parameter
           inclination (float): inclination angle
           roll (float): value for the roll rotations
           ref_vec (numpy array): where the rotation is occurring around
           hinge (numpy array): values of the hinge rotation

       Returns:
            x (numpy array): of models new x coordinates
            y (numpy array): of models new y coordinates
            z (numpy array): of models new z coordinates
            indices (numpy array): valid coordinates ?
    """


    if (np.any(model) == False):
        length = max(np.shape(x_c))
        indices = np.linspace(0,length-1,num=(length),dtype= int)
    else:
        indices = np.argwhere(model)
        indices = indices[:,0]

    R = Rz(heading) @ Ry(inclination) @ Rx(roll)

    new_coor = R @ np.array([x_c[indices] - hinge[0], y_c[indices] - hinge[1], z_c[indices] - hinge[2]])

    X = new_coor[0,:] + hinge[0] + ref_vec[0]
    Y = new_coor[1,:] + hinge[1] + ref_vec[1]
    Z = new_coor[2,:] + hinge[2] + ref_vec[2]

    return np.array(X)[0,:], np.array(Y)[0,:], np.array(Z)[0,:], indices
    return X, Y, Z, indices

def return_head_real_model_new(x,fishlen,proj_params,cb,cs1,cs2):
    """
       Function used to create fish head model

       Args:
           x (numpy array): 22 parameter vector specifying position of fish
           fishlen (float): fish length
           proj_params (string): path to projection parameters to calculate diffraction
           c_b (numpy array): keypoint coordinate in bottom view
           c_s1 (numpy array): keypoint coordinate in side view 1
           c_s2 (numpy array): keypoint coordinate in side view 2
       Returns:
            graymodel_b: (numpy array): head as seen from bottom view
            graymodel_s1: (numpy array): head as seen from side view 1
            graymodel_s2: (numpy array): head as seen from side view 2
            eye_b (numpy array): eye coordinate as seen from bottom view
            eye_s1 (numpy array): eye coordinates as seen from side view 1
            eye_s2 (numpy array): eye coordinates as seen from side view 2
            eye_3d_coor (numpy array): eye coordinates in 3d space
       """

    # Calculate the 3D points pt from model parameters
    seglen = fishlen * 0.09
    size_lut_3d = 2 # Represents the length of the box in which the 3D fish is constructed
    inclination = x[12]
    heading = x[3]
    hp = np.array([[x[0]],[x[1]],[x[2]]])

    dtheta = x[3:12]
    theta = np.cumsum(dtheta)
    dphi = x[12:21]
    phi = np.cumsum(dphi)
    roll = x[21]

    vec_unit = seglen* np.array([[np.cos(theta) * np.cos(phi)], [np.sin(theta) * np.cos(phi)], [-np.sin(phi)]])
    vec_unit = vec_unit[:,0,:]
    # vec_ref_1 is parallel to the camera sensor of b and s2
    # vec_ref_2 is parallel to s1
    vec_ref_1 = np.array([[seglen],[0],[0]])
    vec_ref_2 = np.array([[0],[seglen],[0]])
    pt_ref = np.array([hp + vec_ref_1, hp + vec_ref_2])
    pt_ref = np.transpose(pt_ref[:, :, 0])

    z = np.array([[0], [0], [0]])
    temp = np.concatenate((z, vec_unit), axis=1)
    pt = np.cumsum(temp, axis=1)

    pt = np.concatenate((pt + np.tile(hp, (1, 10)), pt_ref), axis=1)


    # Construct the larva
    # Locate center of the head to use as origin for rotation of the larva.
    # This is consistent with the way in which the parameters of the model are
    # computed during optimization
    resolution = 75
    x_c = (np.linspace(0, 1, num=resolution) * size_lut_3d)
    y_c = (np.linspace(0, 1, num=resolution) * size_lut_3d)
    z_c = (np.linspace(0, 1, num=resolution) * size_lut_3d)

    [x_c,y_c,z_c] = np.meshgrid(x_c,y_c,z_c)

    x_c = x_c.transpose()
    x_c = x_c.flatten()

    y_c = y_c.transpose()
    y_c = y_c.flatten()

    z_c = z_c.transpose()
    z_c = z_c.flatten()

    pt_original = np.zeros((3,3))
    pt_original[:,1] = np.array([size_lut_3d/2, size_lut_3d/2, size_lut_3d/2])
    pt_original[:,0] = pt_original[:,1] - np.array([seglen,0,0])
    pt_original[:, 2] = pt_original[:, 1] + np.array([seglen, 0, 0])


    hinge = pt_original[:,2]

    vec_13 = pt[:,0] - pt[:, 2]
    vec_13 = np.array([[vec_13[0]],[vec_13[1]],[vec_13[2]]])

    vec_13 = np.tile(vec_13,(1,12))

    pt = pt + vec_13

    ref_vec = pt[:,2] - hinge
    eye_br = 13
    head_br = 13
    belly_br = 13

    random_vector_eye = np.random.rand(5)

    [eye1_model, eye1_c] = eye1model(x_c, y_c, z_c, seglen, eye_br, size_lut_3d, random_vector_eye)
    [eye2_model, eye2_c] = eye2model(x_c, y_c, z_c, seglen, eye_br, size_lut_3d, random_vector_eye)

    belly_model = bellymodel(x_c, y_c, z_c, seglen, belly_br, size_lut_3d)

    head_model = headmodel(x_c, y_c, z_c, seglen, head_br, size_lut_3d)

    model_X, model_Y, model_Z, indices = reorient_model(eye1_model, x_c, y_c, z_c, heading, inclination, roll, ref_vec, hinge)

    [eye1_b, eye1_s1, eye1_s2] = project_camera_copy(eye1_model, model_X, model_Y, model_Z, proj_params, indices, cb, cs1, cs2)

    [model_X, model_Y, model_Z, indices] = reorient_model(eye2_model, x_c, y_c, z_c, heading, inclination, roll, ref_vec, hinge)

    [eye2_b, eye2_s1, eye2_s2] = project_camera_copy(eye2_model, model_X, model_Y, model_Z, proj_params, indices, cb, cs1, cs2)

    [model_X, model_Y, model_Z, indices] = reorient_model(head_model, x_c, y_c, z_c, heading, inclination, roll, ref_vec, hinge)

    [head_b, head_s1, head_s2] = project_camera_copy(head_model, model_X, model_Y, model_Z, proj_params, indices, cb, cs1, cs2)

    [model_X, model_Y, model_Z, indices] = reorient_model(belly_model, x_c, y_c, z_c, heading, inclination, roll, ref_vec, hinge)

    [belly_b, belly_s1, belly_s2] = project_camera_copy(belly_model, model_X, model_Y, model_Z, proj_params, indices, cb, cs1, cs2)

    eye_br = 114 + (np.random.rand() - 0.5) * 5
    head_br = 72 + (np.random.rand() - 0.5) * 5
    belly_br = 83 + (np.random.rand() - 0.5) * 5

    # TODO: find a more elegant way to deal with the try and except statements, they are currently in place
    #       to render fishes better at the edges, since otherwise is crashes early
    try:
        eye_scaling = np.maximum(eye_br / float(max(eye1_b.max(axis=0))), eye_br / float(max(eye2_b.max(axis=0))))
    except ZeroDivisionError:
        eye_scaling = 1

    eye1_b = eye1_b * (eye_scaling)
    eye2_b = eye2_b * (eye_scaling)

    try:
        head_b = head_b * (head_br / float(max(head_b.max(axis=0))))
    except ZeroDivisionError:
        throwaway = 99
    try:
        belly_b = belly_b * (belly_br / float(max(belly_b.max(axis=0))))
    except ZeroDivisionError:
        throwaway = 99


    # S1 upperparts, try except blocks kept seperate in case one is drawable
    try:
        eye_scaling = max(eye_br / float(max(eye1_s1.max(axis=0))), eye_br / float(max(eye2_s1.max(axis=0))))
    except ZeroDivisionError:
        #The eye was out of bounds, the eye scalling does not matter
        eye_scaling = 1
    eye1_s1 = eye1_s1 * (eye_scaling)
    try:
        eye2_s1 = eye2_s1 * (eye_br / float(max(eye2_s1.max(axis=0))))
    except ZeroDivisionError:
        #Out of bounds
        throwaway = 99
    try:
        head_s1 = head_s1 * (head_br / float(max(head_s1.max(axis=0))))
    except ZeroDivisionError:
        #Out of bounds
        throwaway = 99
    try:
        belly_s1 = belly_s1 * (belly_br / float(max(belly_s1.max(axis=0))))
    except ZeroDivisionError:
        #Out of bounds
        throwaway = 99


    try:
        eye_scaling = max(eye_br / float(max(eye1_s2.max(axis=0))), eye_br / float(max(eye2_s2.max(axis=0))))
    except ZeroDivisionError:
        #Out of bounds, eye scalling dosent matter since it is out of the picture
        eye_scaling = 1

    try:
        eye1_s2 = eye1_s2 * (eye_scaling)
    except ZeroDivisionError:
        #Out of bounds
        throwaway = 99

    try:
        eye2_s2 = eye2_s2 * (eye_scaling)
    except ZeroDivisionError:
        #Out of bounds
        throwaway = 99

    try:
        head_s2 = head_s2 * (head_br / float(max(head_s2.max(axis=0))))
    except ZeroDivisionError:
        #Out of bounds
        throwaway = 99

    try:
        belly_s2 = belly_s2 * (belly_br / float(max(belly_s2.max(axis=0))))
    except ZeroDivisionError:
        #Out of bounds
        throwaway = 99


    graymodel_b = np.maximum(np.maximum(np.maximum(eye1_b, eye2_b), head_b), belly_b)
    graymodel_s1 = np.maximum(np.maximum(np.maximum(eye1_s1, eye2_s1), head_s1), belly_s1)
    graymodel_s2 = np.maximum(np.maximum(np.maximum(eye1_s2, eye2_s2), head_s2), belly_s2)

    [eyeCenters_X, eyeCenters_Y, eyeCenters_Z, throwaway] = reorient_model(np.array([]), np.array([eye1_c[0], eye2_c[0]]), np.array([eye1_c[1], eye2_c[1]]), np.array([eye1_c[2], eye2_c[2]]), heading, inclination, roll, ref_vec, hinge)


    [eye_b, eye_s1, eye_s2] = calc_proj_w_refra_cpu(np.array([eyeCenters_X, eyeCenters_Y, eyeCenters_Z]), proj_params)

    eye_3d_coor = np.array([eyeCenters_X, eyeCenters_Y, eyeCenters_Z])

    # Original and slightly better version, but not present in Hals venv
    # graymodel_b = (scipy.signal.medfilt2d(graymodel_b))
    # graymodel_s1 = (scipy.signal.medfilt2d(graymodel_s1))
    # graymodel_s2 = (scipy.signal.medfilt2d(graymodel_s2))

    graymodel_b = np.float32(graymodel_b)
    graymodel_b = cv.medianBlur(graymodel_b, 3)
    graymodel_s1 = np.float32(graymodel_s1)
    graymodel_s1 = cv.medianBlur(graymodel_s1,3)
    graymodel_s2 = np.float32(graymodel_s2)
    graymodel_s2 = cv.medianBlur(graymodel_s2,3)

    return graymodel_b, graymodel_s1, graymodel_s2, eye_b, eye_s1, eye_s2, eye_3d_coor

def linIndxTo2DIndx(num, arrShape):
    """
        Function turn linear indices to 2d as done by matlab
        Args:
            num (int): idx
            arrShape (list): dimensions of array
        Returns:
             idices (tuple): row, coloumn
        """
    rows = arrShape[0]

    columnIndx = num / rows
    if np.floor(columnIndx) == columnIndx and columnIndx != 0:
        columnIndx = columnIndx - 1
    else:
        columnIndx = np.floor(columnIndx)

    rowIndx = (num % rows) - 1

    if rowIndx == -1:
        rowIndx = rows - 1

    return (int(rowIndx),int(columnIndx))

def view_b_lut_new_real_cpu(crop_coor, pt, lut_tail, projection, imageSizeX, imageSizeY):
    """
       Function used to create fish model as if seen from bottom view

       Args:
           crop_coor (numpy array): cropping coordinates
           pt (float): position of the fish
           lut_tail (numpy array): look up table for tail
           projection (numpy array): fish head as seen from side view
           imageSizeX (int): width of image
           imageSizeY (int): height of image
       Returns:
            graymodel (numpy array): image from bottom view
       """

    vec_pt = pt[:,1: 10] - pt[:, 0: 9]

    segslen = (np.sum(vec_pt * vec_pt, 0))**(1/2)

    segslen = np.tile(segslen, (2, 1))
    vec_pt_unit = vec_pt / segslen

    theta_prime = np.arctan2(vec_pt_unit[1,:],vec_pt_unit[0,:])
    theta = np.zeros((2,max(theta_prime.shape)))
    theta[0,:] = theta_prime
    theta[1,:] = theta_prime

    #shift pts t0 the cropped images

    pt[0,:] = pt[0,:] - crop_coor[2] + 1

    pt[1,:] = pt[1,:] - crop_coor[0] + 1

    imblank = np.zeros((imageSizeY, imageSizeX))

    bodypix = imblank


    headpix = uint8(uint8(projection / 2) * 5.2)

    size_lut = 19

    size_half = (size_lut + 1) / 2


    coor_t = np.floor(pt)

    dt = np.floor((pt - coor_t) * 5) + 1

    at = np.mod(np.floor(theta * 180 / np.pi), 360) + 1

    seglen = segslen

    indices = np.argwhere(seglen<3.3)
    for index in indices:
        seglen[index[0],index[1]] = 3.2

    indices = np.argwhere(seglen>10.5)
    for index in indices:
        seglen[index[0],index[1]] =10.6


    seglenidx = roundHalfUp((seglen - 5) / 0.2)

    #Uncomment For the fixed version
    seglenidx = seglenidx[0,:]
    for ni in range(0,7):

        n = ni + 2

        tailpix = imblank

        tail_model = gen_lut_b_tail(ni + 1, seglenidx[n], dt[0, n], dt[1, n], at[0,n])
        # newIndex = linIndxTo2DIndx(n+1,seglenidx.shape)
        # tail_model = gen_lut_b_tail.gen_lut_b_tail(ni + 1, seglenidx[newIndex], dt[0, n], dt[1, n], at[0,n])

        try:
            tailpix[int(max(1, coor_t[1, n] - (size_half - 1))) - 1: int(min(imageSizeY, coor_t[1, n] + (size_half - 1))), int(max(1, coor_t[0, n] - (size_half - 1))) - 1: int(min(imageSizeX, coor_t[0, n] + (size_half - 1)))] = tail_model[int(max((size_half + 1) - coor_t[1, n], 1))-1: int(min(imageSizeY - coor_t[1, n] + size_half, size_lut)), int(max((size_half + 1 ) - coor_t[0, n], 1))-1: int(min(imageSizeX - coor_t[0, n] + size_half, size_lut))]
        except ValueError:
            throwaway = 99
        bodypix = np.maximum(bodypix, tailpix)
        bodypix = uint8(bodypix)

    graymodel = np.maximum(headpix, uint8((np.random.normal(0.6, 0.08)) * bodypix))
    graymodel = uint8(graymodel)

    return graymodel

def view_s_lut_new_real_cpu(crop_coor, pt,lut_tail,projection,imageSizeX,imageSizeY):
    """
       Function used to create fish model as if seen from a side view

       Args:
           crop_coor (numpy array): cropping coordinates
           pt (float): position of the fish
           lut_tail (numpy array): look up table for tail
           projection (numpy array): fish head as seen from side view
           imageSizeX (int): width of image
           imageSizeY (int): height of image
       Returns:
            graymodel (numpy array): image from side view
       """

    # Find the coefficients of the line that defines the refracted ray
    vec_pt = pt[:,1:10] - pt[:,0:9]

    segslen = (np.sum(vec_pt*vec_pt,0))**(1/2)

    segslen = np.tile(segslen, (2, 1))

    vec_pt_unit = vec_pt /segslen

    theta_prime = np.arctan2(vec_pt_unit[1, :], vec_pt_unit[0, :])
    theta = np.zeros((2, max(theta_prime.shape)))
    theta[0, :] = theta_prime
    theta[1:] = theta_prime



    # shift pts to the cropped images

    pt[0,:] = pt[0,:] - crop_coor[2] + 1

    pt[1,:] = pt[1,:] - crop_coor[0] + 1


    ###These three are all uint8
    imblank = np.zeros((imageSizeY,imageSizeX))

    imblank_cpu = np.zeros((imageSizeY,imageSizeX))

    bodypix = imblank_cpu

    headpix = uint8(uint8(projection/1.8)*5.2)

    # tail

    size_lut = 15

    size_half = (size_lut+1)/2

    seglen = segslen


    seglen[seglen<0.2] = 0.1



    seglen[seglen>7.9] = 8



    seglenidx = roundHalfUp(seglen/0.2)

    coor_t = np.floor(pt)

    dt = np.floor((pt - coor_t)*5) + 1

    at = np.mod(np.floor(theta*90/np.pi),180) + 1

    seglenidx = seglenidx[0,:]
    for ni in range(0,7):

        n = ni+2

        tailpix = imblank

        #newIndex = linIndxTo2DIndx(n+1, seglenidx.shape)
        tail_model = gen_lut_s_tail(ni + 1, seglenidx[n], dt[0,n], dt[1,n], at[0,n])
        try:
            tailpix[int(max(1, coor_t[1, n] - (size_half - 1))) - 1: int(min(imageSizeY, coor_t[1, n] + (size_half - 1))), int(max(1, coor_t[0, n] - (size_half - 1))) - 1: int(min(imageSizeX, coor_t[0, n] + (size_half - 1)))] = tail_model[int(max((size_half + 1) - coor_t[1, n], 1))-1: int(min(imageSizeY - coor_t[1, n] + size_half, size_lut)), int(max((size_half + 1 ) - coor_t[0, n], 1))-1: int(min(imageSizeX - coor_t[0, n] + size_half, size_lut))]
        except ValueError:
            #Tail went out of bounds, so we do not take those values
            throwaway = 99
        bodypix = np.maximum(bodypix, tailpix)




    graymodel = np.maximum(headpix,(np.random.normal(0.8, 0.05))*bodypix)
    graymodel = uint8(graymodel)

    return graymodel


def return_graymodels_fish(x,lut_b_tail,lut_s_tail,proj_params,fishlen,imageSizeX,imageSizeY):
    """
    Function used to generate a fish defined by the 22 parameter vector x and fishlen

    Args:
        x (numpy array): 22 parameter vector
        lut_b_tail (numpy array): look up table to draw tail from bottom view
        lut_s_tail (numpy array): look up table to draw tail from side view
        proj_params (string): Path to projection parameters to calculate refraction
        fishlen (int): fish length, less than 2 and grater than zero
        imageSizeX (int): width of image to be generated
        imageSizeY (int): height of image to be generated

    Returns:
        gray_b (int): grayscale image of bottom view, size imageSizeX by imageSizeY
        gray_s1 (int): grayscale image of side view 1, size imageSizeX by imageSizeY
        gray_s2 (int): grayscale image of bottom view 2, size imageSizeX by imageSizeY
        crop_b (numpy array): cropping Indices, [smallestY, biggestY, smallestX, biggestX], for bottom view
        crop_s1 (numpy array): cropping Indices, [smallestY, biggestY, smallestX, biggestX], for s1 view
        crop_s2 (numpy array): cropping Indices, [smallestY, biggestY, smallestX, biggestX], for s2 view
        annotated_b (numpy array): array of backbone points in view b, first row represents x and second row y
        annotated_s1 (numpy array): array of backbone points in view s1, first row represents x and second row y
        annotated_s2 (numpy array): array of backbone points in view s2, first row represents x and second row y
        eye_b (numpy array): array of eye points in view b, first row represents x and second row y
        eye_s1 (numpy array): array of eye points in view s1, first row represents x and second row y
        eye_s2 (numpy array): array of eye points in view s2, first row represents x and second row y
        coor_3d (numpy array): keypoints array representing them in actual 3d space
    """

    # Re-defining cropped coordinates for training images of dimensions
    # imageSizeY x imageSizeX
    crop_b = np.array([0, imageSizeY - 1, 0, imageSizeX - 1])
    crop_s1 = np.array([0, imageSizeY - 1, 0, imageSizeX - 1])
    crop_s2 = np.array([0, imageSizeY - 1, 0, imageSizeX - 1])
    crop_b = crop_b + 1
    crop_s1 = crop_s1 + 1
    crop_s2 = crop_s2 + 1

    # initial guess of the position
    # seglen is the length of each segment
    seglen = fishlen*0.09
    # alpha: azimuthal angle of the rotated plane
    # gamma: direction cosine of the plane of the fish with z-axis
    # theta: angles between segments along the plane with direction cosines
    # alpha, beta and gamma
    hp = np.array([[x[0]],[x[1]],[x[2]]])
    dtheta = x[3:12]
    theta = np.cumsum(dtheta)
    dphi = x[12:21]
    phi = np.cumsum(dphi)

    vec = seglen*np.array([np.cos(theta)*np.cos(phi), np.sin(theta) *np.cos(phi), -np.sin(phi)])

    # vec_ref_1 is parallel to the camera sensor of b and s2
    # vec_ref_2 is parallel to s1
    vec_ref_1 = np.array([[seglen],[0],[0]])
    vec_ref_2 = np.array([[0],[seglen],[0]])

    pt_ref = np.concatenate((hp + vec_ref_1, hp + vec_ref_2), axis=1)

    z = np.array([[0], [0], [0]])
    frank = np.concatenate((z, vec), axis=1)
    pt = np.cumsum(frank,axis=1)

    pt = np.concatenate((pt + np.tile(hp, (1, 10)),pt_ref),axis=1)

    # use cen_3d as the 4th point on fish
    hinge = pt[:,2]
    vec_13 = pt[:,0] - hinge
    temp1 = vec_13[0]
    temp2 = vec_13[1]
    temp3 = vec_13[2]
    vec_13 = np.array([[temp1],[temp2],[temp3]])
    vec_13 = np.tile(vec_13, (1, 12))

    pt = pt + vec_13

    [coor_b,coor_s1,coor_s2] = calc_proj_w_refra_cpu(pt, proj_params)

    # keep the corresponding vec_ref for each
    coor_b_shape = np.shape(coor_b)
    coor_b = coor_b[:,0:coor_b_shape[1]-2]
    idxs = [*range(coor_s1.shape[1])]
    idxs.pop(coor_s1.shape[1]-2)  # this removes elements from the list
    coor_s1 = coor_s1[:, idxs]
    coor_s2 = coor_s2[:,0:coor_s2.shape[1]-1]


    (projection_b,projection_s1,projection_s2,eye_b,eye_s1,eye_s2,eye_coor_3d) = return_head_real_model_new(x, fishlen, proj_params, crop_b, crop_s1, crop_s2)

    temp = np.copy(coor_b)
    # TODO: got to do this part more elegant
    try:
        gray_b = view_b_lut_new_real_cpu(crop_b, temp, lut_b_tail, projection_b, imageSizeX, imageSizeY)
    except:
        gray_b = np.zeros((imageSizeY,imageSizeX))

    temp = np.copy(coor_s1)
    try:
        gray_s1 = view_s_lut_new_real_cpu(crop_s1, temp, lut_s_tail, projection_s1, imageSizeX, imageSizeY)
    except:
        gray_s1 = np.zeros((imageSizeY,imageSizeX))

    temp = np.copy(coor_s2)
    try:
        gray_s2 = view_s_lut_new_real_cpu(crop_s2, temp, lut_s_tail, projection_s2, imageSizeX, imageSizeY)
    except:
        gray_s2 = np.zeros((imageSizeY,imageSizeX))


    annotated_b = np.zeros((2,coor_b.shape[1]))
    annotated_b[0,:] = coor_b[0,:] - crop_b[2] + 1
    annotated_b[1,:] = coor_b[1,:] - crop_b[0] + 1

    annotated_s1 = np.zeros((2,coor_s1.shape[1]))
    annotated_s1[0,:] = coor_s1[0,:] - crop_s1[2] + 1
    annotated_s1[1,:] = coor_s1[1,:] - crop_s1[0] + 1

    annotated_s2 = np.zeros((2,coor_s2.shape[1]))
    annotated_s2[0,:] = coor_s2[0,:] - crop_s2[2] + 1
    annotated_s2[1,:] = coor_s2[1,:] - crop_s2[0] + 1

    annotated_b = annotated_b[:,0:10]
    annotated_s1 = annotated_s1[:,0:10]
    annotated_s2 = annotated_s2[:,0:10]

    eye_b[0,:] = eye_b[0,:] - crop_b[2] + 1
    eye_b[1,:] = eye_b[1,:] - crop_b[0] + 1
    eye_s1[0,:] = eye_s1[0,:] - crop_s1[2] + 1
    eye_s1[1,:] = eye_s1[1,:] - crop_s1[0] + 1
    eye_s2[0,:] = eye_s2[0,:] - crop_s2[2] + 1
    eye_s2[1,:] = eye_s2[1,:] - crop_s2[0] + 1

    #Subtract 1 for accordance with python's format
    eye_b = eye_b - 1
    eye_s1 = eye_s1 - 1
    eye_s2 = eye_s2 - 1

    annotated_b = annotated_b - 1
    annotated_s1 = annotated_s1 - 1
    annotated_s2 = annotated_s2 - 1

    crop_b = crop_b - 1
    crop_s1 = crop_s1 - 1
    crop_s2 = crop_s2 - 1

    coor_3d = pt[:,0:10]

    coor_3d = np.concatenate((coor_3d,eye_coor_3d),axis=1)

    return gray_b, gray_s1, gray_s2, crop_b, crop_s1, crop_s2,annotated_b, annotated_s1, annotated_s2, eye_b, eye_s1, eye_s2,coor_3d

# Multiple Fish Functions
def imGaussNoise(image,mean,var):
    """
       Function used to make image have static noise

       Args:
           image (numpy array): image
           mean (float): mean
           var (numpy array): var

       Returns:
            noisy (numpy array): image with noise applied
       """
    row,col= image.shape
    sigma = var**0.5
    gauss = np.random.normal(mean,sigma,(row,col))
    gauss = gauss.reshape(row,col)
    noisy = image + gauss
    return noisy