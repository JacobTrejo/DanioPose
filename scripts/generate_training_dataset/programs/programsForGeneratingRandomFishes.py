import numpy as np
import numpy.ma as ma
import scipy
from scipy.io import loadmat
import cv2 as cv
import math
import copy
from skimage.util import random_noise
from programs.programsForDrawingImage import *

def xTo3DPoints(x,fishlen):
    """
       Computes the backbone points in 3d space from x vector and fishlen

       Args:
           x (numpy array): 22 parameter vector specifying fish pose
           fishlen (float): fish length

       Returns:
            points (numpy array): the 10 backbone points in the fish in 3d
       """
    seglen = fishlen * 0.09
    # alpha: azimuthal angle of the rotated plane
    # gamma: direction cosine of the plane of the fish with z-axis
    # theta: angles between segments along the plane with direction cosines
    # alpha, beta and gamma
    hp = np.array([[x[0]], [x[1]], [x[2]]])
    dtheta = x[3:12]
    theta = np.cumsum(dtheta)
    dphi = x[12:21]
    phi = np.cumsum(dphi)

    vec = seglen * np.array([np.cos(theta) * np.cos(phi), np.sin(theta) * np.cos(phi), -np.sin(phi)])

    # vec_ref_1 is parallel to the camera sensor of b and s2
    # vec_ref_2 is parallel to s1
    vec_ref_1 = np.array([[seglen], [0], [0]])
    vec_ref_2 = np.array([[0], [seglen], [0]])

    pt_ref = np.concatenate((hp + vec_ref_1, hp + vec_ref_2), axis=1)

    z = np.array([[0], [0], [0]])
    frank = np.concatenate((z, vec), axis=1)
    pt = np.cumsum(frank, axis=1)

    pt = np.concatenate((pt + np.tile(hp, (1, 10)), pt_ref), axis=1)

    hinge = pt[:,2]
    vec_13 = pt[:,0] - hinge
    temp1 = vec_13[0]
    temp2 = vec_13[1]
    temp3 = vec_13[2]
    vec_13 = np.array([[temp1],[temp2],[temp3]])
    vec_13 = np.tile(vec_13, (1, 12))

    pt = pt + vec_13

    return pt[:,:-2]


def newAndImproveXTo3DPoints(x, fishlen):
    """
       Computes the backbone points in 3d space from x vector and fishlen,
       this time taking into consideration the 3rd rotation variable

       Args:
           x (numpy array): 22 parameter vector specifying fish pose
           fishlen (float): fish length

       Returns:
           points (numpy array): backbone points in 3d space
    """

    seglen = fishlen * 0.09
    hp = np.array([[x[0]], [x[1]], [x[2]]])
    dtheta = x[4:12]

    theta = np.cumsum(np.concatenate(([0], dtheta)))
    dphi = x[13:21]
    phi = np.cumsum(np.concatenate(([0], dphi)))

    vec = seglen * np.array([np.cos(theta) * np.cos(phi), np.sin(theta) * np.cos(phi), -np.sin(phi)])

    theta_0 = x[3]
    phi_0 = x[12]
    gamma_0 = x[21]

    vec_ref_1 = np.array([[seglen], [0], [0]])
    vec_ref_2 = np.array([[0], [seglen], [0]])

    pt_ref = np.concatenate((hp + vec_ref_1, hp + vec_ref_2), axis=1)

    z = np.array([[0], [0], [0]])
    frank = np.concatenate((z, vec), axis=1)
    pt = np.cumsum(frank, axis=1)

    pt = Rz(theta_0) @ Ry(phi_0) @ Rx(gamma_0) @ pt

    pt = np.array(pt)
    pt = np.concatenate((pt + np.tile(hp, (1, 10)), pt_ref), axis=1)

    hinge = pt[:, 2]
    vec_13 = pt[:, 0] - hinge

    vec_13 = np.array([[vec_13[0]], [vec_13[1]], [vec_13[2]]])
    vec_13 = np.tile(vec_13, (1, 12))

    pt = pt + vec_13
    pt = np.array(pt)

    return pt[:, :-2]

def addBoxes(points3d,padding = .7):
    """
        Function to add boxes to backbone points, to stop fishes from phasing through each other

        Args:
            points3d (numpy array): 10 by 3 vector specifying the backbone positions in 3d
            padding (float, optional): how much padding the add to the boxes, by a factor of (1 + padding)

        Returns:
            pointsAndBoxes (numpy array): 9 by to array in which the first 3 rows are the x, y, and z coordinates
            of the keypoints, the next three rows are the center coordinates of the bounding box, and the final
            three rows are the lengths in the respective coordinates of said bounding boxes
    """
    pointsAndBoxes = np.zeros((9,10))
    pointsAndBoxes[0:3,:] =points3d
    for pointIdx in range(9):

        (bX, bY, bZ) = (points3d[0, pointIdx], points3d[1, pointIdx], points3d[ 2, pointIdx])
        (aX, aY, aZ) = (points3d[0, pointIdx+1], points3d[1, pointIdx+1], points3d[ 2, pointIdx+1])

        t = np.array([max(bX, aX), max(bY, aY), max(bZ, aZ)])
        l = np.array([np.abs(aX - bX), np.abs(aY - bY), np.abs(aZ - bZ)])

        c = t - l / 2

        l[l <= .06] = .1
        l = (padding + 1) * l

        if pointIdx == 0:
            l = l * 1.5

        pointsAndBoxes[3:, pointIdx] = [c[0], c[1], c[2], l[0], l[1], l[2]]
    return pointsAndBoxes



def isFishWithinBounds(fishX, fishlen):
    """
        Quick function to check if the fish is in the image

        Args:
            fishX (numpy array): 22 parameter vector
            fishlen (float): fish length

        Returns:
            TrueOrFalse (boolean): True or False
    """
    imagesizeY = 488
    imagesizeX = 648

    # # crops generated from the com being at (0,0,72.5)
    # crop_b = np.array([-12.5, 474.5, -6.5, 640.5])
    # crop_s1 = np.array([-26.5, 460.5, 12.5, 659.5])
    # crop_s2 = np.array([-22.5, 464.5, -8.5, 638.5])
    global crop_b
    global crop_s1
    global crop_s2

    fish3DPoints = xTo3DPoints(fishX, fishlen)
    seglen = fishlen * .09

    # It's possible that I can just check for the top and bottom corner of the 'box' around the points

    # this is the offset that wll be added/subtracted from each backbone
    # fish to ensure that most of the fish is withing the image
    offset = np.ones(10) * seglen * 2
    # A wider space for the head
    offset[0:2] = offset[0:2] * 2

    pointsWithPadding = np.zeros((6, 3, 10))
    for x in range(6):
        pointsWithPadding[x] = np.copy(fish3DPoints)
        if x < 3:
            pointsWithPadding[x][x, :] += offset
        else:
            pointsWithPadding[x][x - 3, :] -= offset

    for x in range(6):
        [pb, ps1, ps2] = calc_proj_w_refra_cpu(pointsWithPadding[x])

        # For reference whether they represent rows or columns
        # gray_b[(np.ceil(pb[1,:]).astype(int)),(np.ceil(pb[0,:]).astype(int))] = -90
        # gray_s2[(np.ceil(ps2[1, :]).astype(int)), (np.ceil(ps2[0, :]).astype(int))] = -90
        # gray_s1[(np.ceil(ps1[1, :]).astype(int)), (np.ceil(ps1[0, :]).astype(int))] = -90
        ps2[0, :] = ps2[0, :] - crop_s2[2]
        ps2[1, :] = ps2[1, :] - crop_s2[0]

        ps1[0, :] = ps1[0, :] - crop_s1[2]
        ps1[1, :] = ps1[1, :] - crop_s1[0]

        pb[0, :] = pb[0, :] - crop_b[2]
        pb[1, :] = pb[1, :] - crop_b[0]
        # TODO Rename
        # Making sure the backbones are not out of bounds
        if np.any(pb[1, :] < 0)  or np.any(pb[1, :] >= 487)  \
                or np.any(pb[0, :] < 0)  or np.any(pb[0, :] >= 647)  \
                or np.any(ps1[1, :] < 0) or np.any(ps1[1, :] >= 487) \
                or np.any(ps1[0, :] < 0) or np.any(ps1[0, :] >= 647) \
                or np.any(ps2[1, :] < 0) or np.any(ps2[1, :] >= 487) \
                or np.any(ps2[0, :] < 0) or np.any(ps2[0, :] >= 647):
            #Fish is out of bounds
            return False
    #Fish is within bounds
    return True

def doesThisFishInterfere(fish,fishes):
    """
        Quick function to check if the fish is going to phase through any of the other fishes

        Args:
            fish (numpy array): fishlen + 22 parameter vector in an array
            fishes (numpy array): array of fishes in which each row holds a 23 parameter vector
                                    the first value is fishlen and the rest is the 22 parameter vector for the fish
        Returns:
            TrueOrFalse (boolean): True or False
    """
    padding = .7
    simpleFishShellArr = np.zeros((fishes.shape[0]+1,9,10))

    newFishPoints = xTo3DPoints(fish[1:],fish[0])
    newFishPointsAndBoxes = addBoxes(newFishPoints)

    simpleFishShellArr[0,:,:] = newFishPointsAndBoxes

    idx = 1
    for goodFish in fishes:
        goodFishPoints = xTo3DPoints(goodFish[1:],goodFish[0])
        goodFishPointsAndBoxes = addBoxes(goodFishPoints)

        simpleFishShellArr[idx,:,:] = goodFishPointsAndBoxes
        idx +=1

    fishInConsideration = simpleFishShellArr[0]
    for pointNum in range(9):
        c = np.copy(fishInConsideration[ 3:6, pointNum])
        l = np.copy(fishInConsideration[ 6:9, pointNum])
        pMax = c + l / 2
        pMin = c - l / 2

        for fish in simpleFishShellArr[1:]:
            for otherPointNum in range(9):
                oL = np.copy(fish[ 6:9, otherPointNum])
                oC = np.copy(fish[ 3:6, otherPointNum])

                oPMax = oC + oL / 2
                oPMin = oC - oL / 2

                if np.all(pMax > oPMin) and np.all(pMin < oPMax):
                    #It was a bad fish :'(
                    return True
    #Good Fish :)
    return False

imageSizeY = 488
imageSizeX = 648



def createDepthArr(img, xIdx, yIdx, d):
    """
        Gives each pixel of the image depth, it simpy dilates the depth at each keypoint

        Args:
            img (numpy array): img of size imageSizeX by imageSizeY of the fish
            xIdx (numpy array): x coordinates of the keypoints
            yIdx (numpy array): y coordinates of the keypoints
            d (numpy array): the depth of each keypoint
        Returns:
            depthImage (numpy array): img of size imageSizeX by imageSizeY with each pixel of the fish
                                        representing its depth
    """
    depthArr = np.zeros(img.shape)
    depthArrCutOut = np.zeros(img.shape)

    radius = 14
    for point in range(10):
        [backboneY, backboneX] = [(np.ceil(xIdx).astype(int))[point], (np.ceil(yIdx).astype(int))[point]]
        depth = d[point]
        if (backboneY <= imageSizeY-1) and (backboneX <= imageSizeX-1) and (backboneX >= 0) and (backboneY >= 0):
            depthArr[backboneY,backboneX] = depth
    kernel = np.ones(( (radius * 2) + 1, (radius * 2) + 1 ) )
    depthArr = cv.dilate(depthArr,kernel= kernel)

    depthArrCutOut[img != 0] = depthArr[img != 0]
    return depthArrCutOut

def mergeGreysExactly(grays, depths):
    """
        Function that merges grayscale images without blurring them
    :param grays: numpy array of size( n_fishes, imageSizeY, imageSizeX)
    :param depths: numpy array of size( n_fishes, imageSizeY, imageSizeX)
    :return: 2 numpy arrays of size (imageSizeY, imageSizeX) representing the merged depths and grayscale images
        also returns the indices of the fishes in the front
    """
    indicesForTwoAxis = np.indices(grays.shape[1:])

    # indicesFor3dAxis = np.argmin(ma.masked_where(depths == 0, depths), axis=0)
    # has to be masked so that you do not consider parts where there are only zeros
    indicesFor3dAxis = np.argmin(ma.masked_where( grays == 0, depths  ), axis=0 )

    indices2 = indicesFor3dAxis, indicesForTwoAxis[0], indicesForTwoAxis[1]

    mergedGrays = grays[indices2]
    mergedDepths = depths[ indices2]

    return mergedGrays, mergedDepths , indices2

def mergeGreys(grays, depths):
    """
        Function that merges grayscale images while also blurring the edges for a more realistic look
    :param grays: numpy array of size( n_fishes, imageSizeY, imageSizeX)
    :param depths: numpy array of size( n_fishes, imageSizeY, imageSizeX)
    :return: 2 numpy arrays of size (imageSizeY, imageSizeX) representing the merged depths and grayscale images
    """

    # Checking for special cases
    amountOfFishes = grays.shape[0]
    if amountOfFishes == 1:
        return grays[0], depths[0]
    if amountOfFishes == 0 :
        return np.zeros((grays.shape[1:3])), np.zeros((grays.shape[1:3]))

    threshold = 25
    mergedGrays, mergedDepths, indices = mergeGreysExactly(grays, depths)

    # Blurring the edges

    # will be used as the brightness when there is no fish underneath the edges with
    # brightness greater than the threshold
    maxes = np.max(grays, axis=0)

    # will be used as the ordered version of brightnesses greater than the threshold
    grays[grays < threshold] = 0
    graysBiggerThanThresholdMerged, _, _ = mergeGreysExactly(grays, depths)

    # applying the values to the edges
    indicesToBlurr = np.logical_and( np.logical_and( mergedGrays < threshold, mergedGrays > 0 ),
                                     graysBiggerThanThresholdMerged > 0 )
    mergedGrays[ indicesToBlurr ] = graysBiggerThanThresholdMerged[ indicesToBlurr ]
    indicesToBlurr = np.logical_and( np.logical_and( mergedGrays < threshold, mergedGrays > 0 ),
                                     maxes > 0)
    mergedGrays[ indicesToBlurr ] = maxes[indicesToBlurr]

    # NOTE: we could technically also blurr the depths?
    return mergedGrays, mergedDepths



# Inputs
fish_shapes = 'generated_pose_100_percent.mat'
x_all_data = loadmat( inputsFolder +  fish_shapes)
x_all_data = x_all_data['generated_pose']
x_all_data = np.array(x_all_data)
pi = np.pi


#TODO place idx generator inside the function
def generateRandomFish(idx):
    x = np.copy(x_all_data[idx - 1, :])
    x[3] = np.random.rand() * 2 * pi
    x[12] = x_all_data[idx - 1, 17 - 1]
    x[0] = (np.random.rand() - 0.5) * 40
    x[1] = (np.random.rand() - 0.5) * 40
    x[2] = (np.random.rand() - 0.5) * 35 + 72.5
    x[3] = np.random.rand() * 2 * pi
    x[4: 12] = x_all_data[idx - 1, 0: 8]
    x = x[0:13]
    temp = 99
    temp2 = np.concatenate((x, x_all_data[idx - 1, 8: 16], [temp]), axis=0)
    temp2[21] = x_all_data[idx - 1, 18 - 1]
    x = temp2
    fishlen = np.random.normal(3.8, 0.15)

    x[3] = np.random.rand() * 2 * pi
    return x, fishlen


#TODO should it be modified so as to get rid of pvar
def generateRandomFishInView(view ):
    """
        Function that generates a fish that is only visible in a particular view

        Args:
            view (string): either 'B', 'S1', or 'S2'
        Returns:
            x (numpy array): 22 parameter depicting the fish position
            fishlen (float): length of fish
    """

    while True:
        idx = np.random.randint(500000)
        x = np.copy(x_all_data[idx - 1, :])
        x[3] = np.random.rand() * 2 * pi
        x[12] = x_all_data[idx - 1, 17 - 1]

        if view == 'B':
            #modified to stay visible in B
            x[0] = (np.random.rand() - 0.5) * 36
            x[1] = (np.random.rand() - 0.5) * 17

            upOrDown = np.random.rand()
            if upOrDown > .5:
                x[2] = np.random.rand() * 10 + 89
            else:
                x[2] = np.random.rand() * 10 + 45
        if view == 'S1':
            #modified to stay visible in S1
            backOrForward = np.random.rand()
            if backOrForward < .5:
                x[0] = np.random.rand() * 5 + 25
            else:
                x[0] = -22 - (np.random.rand() * 5)

            x[1] = (np.random.rand() * 44) - 21
            x[2] = (np.random.rand() * 33) + 56
        if view == 'S2':
            #modified to stay visible in S2
            x[0] = (np.random.rand() * 45) - 22
            lefOrRight = np.random.rand()
            if lefOrRight > .5:
                x[1] = np.random.rand() * 7 + 25
            else:
                x[1] = -23 - (np.random.rand() * 7)
            x[2] = (np.random.rand() * 34) + 55


        x[3] = np.random.rand() * 2 * pi
        x[4: 12] = x_all_data[idx - 1, 0: 8]
        x = x[0:13]
        temp = 99
        temp2 = np.concatenate((x, x_all_data[idx - 1, 8: 16], [temp]), axis=0)
        temp2[21] = x_all_data[idx - 1, 18 - 1]
        x = temp2
        fishlen = np.random.normal(3.8, 0.15)

        x[3] = np.random.rand() * 2 * pi


        #TODO even better implementation is possible

        [pb, ps1, ps2] = calc_proj_w_refra_cpu( xTo3DPoints(x,fishlen) )

        # # crop_b = np.array([-12.5, 474.5, -6.5, 640.5])
        # # crop_s1 = np.array([-26.5, 460.5, 12.5, 659.5])
        # # crop_s2 = np.array([-22.5, 464.5, -8.5, 638.5])
        # global crop_b
        # global crop_s1
        # global crop_s2
        #
        # ps2[0, :] = ps2[0, :] - crop_s2[2]
        # ps2[1, :] = ps2[1, :] - crop_s2[0]
        # ps1[0, :] = ps1[0, :] - crop_s1[2]
        # ps1[1, :] = ps1[1, :] - crop_s1[0]
        # pb[0, :] = pb[0, :] - crop_b[2]
        # pb[1, :] = pb[1, :] - crop_b[0]

        # possible for better perfection, but such thing should also change the pvar thing
        # and deal more with finding how many of the backbone points are visible

        def areAllPointsOutBounds( pt ):
            isXOutOfBounds = np.all(np.logical_or(pt[0, :] >= 648, pt[0, :] < 0))
            isYOutOfBounds = np.all(np.logical_or(pt[1, :] >= 488, pt[1, :] < 0))
            #replecable with or
            return np.any([isXOutOfBounds, isYOutOfBounds])

        #not exactly the opposite of the above
        def areAllPointsInBounds( pt ):
            isXInBounds = np.all(np.logical_and(pt[0, :] <= 648, pt[0, :] >= 0))
            isYInBounds = np.all(np.logical_and(pt[1, :] <= 488, pt[1, :] >= 0))
            #replecable with and
            return np.all([isXInBounds, isYInBounds])

        if view == 'B':
            if areAllPointsOutBounds(ps1) and areAllPointsOutBounds(ps2) and areAllPointsInBounds(pb):
                break
        if view == 'S1':
            if areAllPointsOutBounds(pb) and areAllPointsOutBounds(ps2) and areAllPointsInBounds(ps1):
                break
        if view == 'S2':
            if areAllPointsOutBounds(pb) and areAllPointsOutBounds(ps1) and areAllPointsInBounds(ps2):
                break

    return x, fishlen

#################################################


def getFishOverlapping(fish, aquarium):
    """
        Function that returns a fish that overlaps the desired fish

        Args:
            fish (numpy array): fishlen + 22 parameter vector for fish pose
            aquarium (numpy array): 23 parameter vectors depicting fishlen + 22 parameter vector,
                                    necessary to ensure fish does not phase through other fishes
        Returns:
            x (numpy array): 22 parameter depicting the fish position
            fishlen (float): length of fish
    """
    viewChoice = np.random.randint(0, 3)
    views = ['B','S1','S2']
    view = views[int(viewChoice)]

    def generateFishAround(fx, fy, fz, view):
        while True:
            idx = np.random.randint(500000)
            x = np.copy(x_all_data[idx - 1, :])
            x[3] = np.random.rand() * 2 * pi
            x[12] = x_all_data[idx - 1, 17 - 1]

            if view == 'B':
                # modified to stay visible in B
                x[0] = fx + (np.random.rand()-.5) * 2
                x[1] = fy + (np.random.rand()-.5) * 2

                x[2] = (np.random.rand() * 33) + 56

            if view == 'S1':
                # modified to stay visible in S1
                x[0] = (np.random.rand() - 0.5) * 36

                x[1] = fy + (np.random.rand()-.5) * 2
                x[2] = fz + (np.random.rand()-.5) * 2
            if view == 'S2':
                # modified to stay visible in S2
                x[0] = fx + (np.random.rand()-.5) * 2
                x[1] = (np.random.rand() - 0.5) * 17
                x[2] = fz + (np.random.rand()-.5) * 2

            x[3] = np.random.rand() * 2 * pi
            x[4: 12] = x_all_data[idx - 1, 0: 8]
            x = x[0:13]
            temp = 99
            temp2 = np.concatenate((x, x_all_data[idx - 1, 8: 16], [temp]), axis=0)
            temp2[21] = x_all_data[idx - 1, 18 - 1]
            x = temp2
            fishlen = np.random.normal(3.8, 0.15)

            x[3] = np.random.rand() * 2 * pi

            # TODO even better implementation is possible
            # TODO should use a different function
            [pb, ps1, ps2] = calc_proj_w_refra_cpu(xTo3DPoints(x, fishlen)  )

            # crop_b = np.array([-12.5, 474.5, -6.5, 640.5])
            # crop_s1 = np.array([-26.5, 460.5, 12.5, 659.5])
            # crop_s2 = np.array([-22.5, 464.5, -8.5, 638.5])
            global crop_b
            global crop_s1
            global crop_s2

            ps2[0, :] = ps2[0, :] - crop_s2[2]
            ps2[1, :] = ps2[1, :] - crop_s2[0]
            ps1[0, :] = ps1[0, :] - crop_s1[2]
            ps1[1, :] = ps1[1, :] - crop_s1[0]
            pb[0, :] = pb[0, :] - crop_b[2]
            pb[1, :] = pb[1, :] - crop_b[0]

            # possible for perfection, but such thing should also change the pvar thing
            # and deal more with finding how many of the backbone points are visible
            fishInFormat = np.zeros(23)
            fishInFormat[0] = fishlen
            fishInFormat[1:] = x
            if doesThisFishInterfere(fishInFormat,aquarium):
                continue

            def areAllPointsOutBounds(pt):
                isXOutOfBounds = np.all(np.logical_or(pt[0, :] >= 648, pt[0, :] < 0))
                isYOutOfBounds = np.all(np.logical_or(pt[1, :] >= 488, pt[1, :] < 0))
                return np.any([isXOutOfBounds, isYOutOfBounds])

            # not exactly the opposite of the above
            def areAllPointsInBounds(pt):
                isXInBounds = np.all(np.logical_and(pt[0, :] <= 648, pt[0, :] >= 0))
                isYInBounds = np.all(np.logical_and(pt[1, :] <= 488, pt[1, :] >= 0))
                return np.all([isXInBounds, isYInBounds])

            if areAllPointsInBounds(pb) and areAllPointsInBounds(ps1) and areAllPointsInBounds(ps2):
                break

        return x, fishlen


    [fx, fy, fz] = fish[1:4]
    x, fishlen = generateFishAround(fx,fy,fz, view)

    return x, fishlen

############################################


def getFishOnEdges(aquarium):
    """
        Function that returns a fish on the edge of a random view

        Args:
            aquarium (numpy array): 23 parameter vectors depicting fishlen + usual 22 parameter vector,
                                    necessary to ensure fish does not phase through other fishes
        Returns:
            x (numpy array): 22 parameter depicting the fish position
            fishlen (float): length of fish
    """
    viewChoice = np.random.randint(0, 3)
    views = ['B','S1','S2']
    view = views[int(viewChoice)]

    def generateFishOnEdge(view):
        while True:
            idx = np.random.randint(500000)
            x = np.copy(x_all_data[idx - 1, :])
            x[3] = np.random.rand() * 2 * pi
            x[12] = x_all_data[idx - 1, 17 - 1]

            verticalOrHorizontal = np.random.rand()
            leftOrRight = np.random.rand()

            if view == 'B':
                if verticalOrHorizontal > .5:
                    if leftOrRight > .5:
                        x[0] = (np.random.rand() - .5) * 36
                        x[1] = (np.random.rand() * -5) - 11
                    else:
                        x[0] = (np.random.rand() - .5) * 36
                        x[1] = (np.random.rand() * 5) + 12
                else:
                    if leftOrRight >.5:
                        x[0] = (np.random.rand() * 5) + 15
                        x[1] = (np.random.rand() - .5) * 28
                    else:
                        x[0] = (np.random.rand() * -5) - 15
                        x[1] = (np.random.rand() - .5) * 28
                x[2] = (np.random.rand() * 36) + 54

            if view == 'S1':
                # modified to stay visible in S1
                x[0] = (np.random.rand() - 0.5) * 36
                if verticalOrHorizontal > .5:
                    if leftOrRight >.5:
                        x[1] = (np.random.rand() * 5) + 21
                        x[2] = (np.random.rand() * 36) + 54
                    else:
                        x[1] = (np.random.rand() * -5) -21
                        x[2] = (np.random.rand() * 36) + 54
                else:
                    if leftOrRight > .5:
                        x[1] = (np.random.rand() -.5) * 46
                        x[2] = (np.random.rand() * 5) + 87
                    else:
                        x[1] = (np.random.rand() -.5) * 46
                        x[2] = (np.random.rand() * -5) + 57

            if view == 'S2':
                x[1] = (np.random.rand() - .5) * 26
                if verticalOrHorizontal > .5:
                    if leftOrRight >.5:
                        x[0] = (np.random.rand() * 5) + 21
                        x[2] = (np.random.rand() * 36) + 54
                    else:
                        x[0] = (np.random.rand() * -5) - 21
                        x[2] = (np.random.rand() * 36) + 54
                else:
                    if leftOrRight > .5:
                        x[0] = (np.random.rand() - .5) * 46
                        x[2] = (np.random.rand() * 5) + 87
                    else:
                        x[0] = (np.random.rand() - .5) * 46
                        x[2] = (np.random.rand() * -5) + 57


            x[3] = np.random.rand() * 2 * pi
            x[4: 12] = x_all_data[idx - 1, 0: 8]
            x = x[0:13]
            temp = 99
            temp2 = np.concatenate((x, x_all_data[idx - 1, 8: 16], [temp]), axis=0)
            # x[13: 21] = x_all_data[idx-1, 8: 16]
            temp2[21] = x_all_data[idx - 1, 18 - 1]
            x = temp2
            fishlen = np.random.normal(3.8, 0.15)

            x[3] = np.random.rand() * 2 * pi

            # TODO even better implementation is possible
            [pb, ps1, ps2] = calc_proj_w_refra_cpu(xTo3DPoints(x, fishlen))

            # crop_b = np.array([-12.5, 474.5, -6.5, 640.5])
            # crop_s1 = np.array([-26.5, 460.5, 12.5, 659.5])
            # crop_s2 = np.array([-22.5, 464.5, -8.5, 638.5])
            global crop_b
            global crop_s1
            global crop_s2

            ps2[0, :] = ps2[0, :] - crop_s2[2]
            ps2[1, :] = ps2[1, :] - crop_s2[0]
            ps1[0, :] = ps1[0, :] - crop_s1[2]
            ps1[1, :] = ps1[1, :] - crop_s1[0]
            pb[0, :] = pb[0, :] - crop_b[2]
            pb[1, :] = pb[1, :] - crop_b[0]

            # possible for better perfection, but such thing should also change the pvar thing
            # and deal more with finding how many of the backbone points are visible
            fishInFormat = np.zeros(23)
            fishInFormat[0] = fishlen
            fishInFormat[1:] = x

            if len(aquarium) != 0:
                if doesThisFishInterfere(fishInFormat,aquarium):
                    continue

            def isAPointInAndOneOut(pt):
                isOneInOneOut = False

                isThereAPointInside = False
                valCIDX = np.logical_and(pt[0, :] >= 0, pt[0, :] < imageSizeX)
                ptR = pt[1, :][valCIDX]
                if np.any(np.logical_and(ptR >= 0, ptR < imageSizeY)):
                    isThereAPointInside = True

                isThereAPointOutside = False
                valCIDX = np.logical_or(pt[0, :] < 0, pt[0, :] >= imageSizeX)
                valRIDX = np.logical_or(pt[1, :] < 0, pt[1, :] >= imageSizeY)
                valOR = np.logical_or(valCIDX,valRIDX)
                if np.any(valOR):
                    isThereAPointOutside = True

                if isThereAPointOutside and isThereAPointInside:
                    isOneInOneOut = True
                return isOneInOneOut

            if view == 'B':
                if isAPointInAndOneOut(pb):
                    break
            if view == 'S1':
                if isAPointInAndOneOut(ps1):
                    break
            if view == 'S2':
                if isAPointInAndOneOut(ps2):
                    break

        return x, fishlen


    x, fishlen = generateFishOnEdge(view)

    return x, fishlen



def generateAquariumOverlapsAndWithoutBounds(amountOfFish, fishesInB, fishesInS1, fishesInS2, overlaping):
    """
        Function that calls generateRandomFish, getFishOverlapping, generateRandomFishInView in order to
        generate an aquarium that satisfies the args.

        Args:
            amountOfFish (int): amount of fishes that are in all views
            fishesInB (int): amount of fishes that are in all views
            fishesInS1 (int): amount of fishes that are in all views
            fishesInS2 (int): amount of fishes that are in all views
            overlapping (int): amount of fishes that are overlapping

        Returns:
            aquarium (numpy array): array of size 23 by totalAmountOfFishes, where each row is a vector of size 23
                                    in which the first value is the fishlen and the rest is the usual 22 parameter
                                    vector
    """
    overlapingCopy = overlaping
    aquarium = np.zeros((amountOfFish + fishesInB + fishesInS1 + fishesInS2 + overlaping, 23))

    # Fish visible in all views
    fishIdx = 0
    for fishNum in range(amountOfFish):
        while True:
            testFish, fishlen = generateRandomFish(np.random.randint(1,500000))
            if fishNum == 0:
                if isFishWithinBounds(testFish, fishlen):
                    aquarium[fishIdx,0] = fishlen
                    aquarium[fishIdx,1:23] = testFish
                    fishIdx += 1

                    if overlaping>0:
                        overlaping -= 1
                        fish, fishlen = getFishOverlapping(aquarium[fishIdx - 1,:],aquarium[:fishIdx,:])
                        fishInFormat = np.zeros((23))
                        fishInFormat[0] = fishlen
                        fishInFormat[1:23] = fish
                        aquarium[fishIdx,:] = fishInFormat
                        fishIdx += 1
                    break
            else:
                testFishInFormat = np.zeros(23)
                testFishInFormat[0] = fishlen
                testFishInFormat[1:] = testFish
                if isFishWithinBounds(testFish,fishlen) and \
                    (not doesThisFishInterfere(testFishInFormat,aquarium[0:fishIdx,:])):
                    aquarium[fishIdx,0] = fishlen
                    aquarium[fishIdx,1:23] = testFish
                    fishIdx += 1

                    if overlaping>0:
                        overlaping -= 1
                        fish, fishlen = getFishOverlapping(aquarium[fishIdx - 1,:],aquarium[:fishIdx,:])
                        fishInFormat = np.zeros((23))
                        fishInFormat[0] = fishlen
                        fishInFormat[1:23] = fish
                        aquarium[fishIdx,:] = fishInFormat
                        fishIdx += 1

                    break

    # Only Visible in certain views
    viewsStrings = ['B','S1','S2']
    fishesInVews = [fishesInB,fishesInS1,fishesInS2]
    fishesCount = [amountOfFish + overlapingCopy, amountOfFish + fishesInB + overlapingCopy, amountOfFish + fishesInB + fishesInS1 + overlapingCopy]
    for viewIdx in range(3):
        for fishNum in range(fishesInVews[viewIdx]):
            while True:
                testFish, fishlen = generateRandomFishInView(viewsStrings[viewIdx])
                if fishesCount[viewIdx] + fishNum == 0:
                    aquarium[0,0] = fishlen
                    aquarium[0,1:23] = testFish
                    break

                else:
                    testFishInFormat = np.zeros(23)
                    testFishInFormat[0] = fishlen
                    testFishInFormat[1:] = testFish
                    if not doesThisFishInterfere(testFishInFormat,aquarium[0:fishNum + fishesCount[viewIdx],:]):
                        aquarium[fishNum + fishesCount[viewIdx],0] = fishlen
                        aquarium[fishNum + fishesCount[viewIdx],1:23] = testFish
                        break



    return aquarium

def generateAquariumOverlapsAndWithoutBoundsAndEdges(amountOfFish, fishesInB, fishesInS1,
                                                     fishesInS2, overlaping, fishesInEdges):
    """
        Function that calls generateAquariumOverlapsAndWithoutBounds and then adds fishes that are in the edges of
        the camera views

        Args:
            amountOfFish (int): amount of fishes that are in all views
            fishesInB (int): amount of fishes that are in all views
            fishesInS1 (int): amount of fishes that are in all views
            fishesInS2 (int): amount of fishes that are in all views
            overlapping (int): amount of fishes that are overlapping
            fishesInEdges (int): amount of fishes that should appear in the edges of the camera views

        Returns:
            aquarium (numpy array): array of size 23 by totalAmountOfFishes, where each row is a vector of size 23
                                    in which the first value is the fishlen and the rest is the usual 22 parameter
                                    vector
    """

    aquarium = generateAquariumOverlapsAndWithoutBounds(amountOfFish,fishesInB,fishesInS1,fishesInS2,overlaping)

    for fishNum in range(fishesInEdges):
        fish, fishlen = getFishOnEdges(aquarium)
        newAquarium = np.zeros((aquarium.shape[0] + 1,aquarium.shape[1]))
        newAquarium[:-1,:] = aquarium
        fishInFormat = np.zeros((23))
        fishInFormat[0] = fishlen
        fishInFormat[1:] = fish
        newAquarium[-1,:] = fishInFormat
        aquarium = np.copy(newAquarium)
    return aquarium