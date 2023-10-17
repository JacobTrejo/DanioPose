import numpy as np
import scipy
from scipy.io import loadmat
import cv2 as cv
import math
import copy
from skimage.util import random_noise
# from programs.programsForDrawingImage import *
from programs.programsForGeneratingRandomFishes import *
from programs.programsForDrawingImageNoRandom import *
class AnnotationsType:
    """
        Static class that holds flags for the types of annotations

        Properties:
            segmentation (string): flag that specifies you want segmentation annotations
            keypoint (string): flag the specifies you want keypoint annotations
    """
    segmentation = 'segmentation'
    keypoint = 'keypoint'
    # Potential for resnet annotations
    # or annotations in .npy files instead of text

class Bounding_Box:
    """
        Class that represents the bounding box of the fish

        Static Properties:
            areaThreshold: value representing the minimal area that a bounding box can have in order to be considered
                            a fish that can hold annotations, used for ignoring barely visible fish in the edges aswell
                            as adding patchy noise
        Properties:
            smallX (int): smallest x value of the bounding box
            smallY (int): smallest y value of the bounding box
            bigX (int): biggest x value of the bounding box
            bigY (int): biggest y value of the bounding box

        Methods:
            __init__, args(int, int, int, int): creates an object representing the bounding box of the fish,
                                                given the smallest x value of the box, the smallest y value of the box
                                                the biggest x value of the box, and the biggest Y value of the box

            getCenterX(): returns the x coordinate of the center of teh bounding box
            getCenterY(): returns y coordinate of the center of the bounding box
            getWidth(): returns the width of the bounding box
            getHeight(): returns height of the bounding box
            getArea(): returns the area of the bounding box
            isBigEnough(): returns the True or False depending on whether the area of the bounding box is
                            greater that the threshold
    """
    areaThreshold = 25

    def __init__(self, smallX = 0, smallY = 0, bigX = 0, bigY = 0):
        self.smallX = smallX
        self.smallY = smallY
        self.bigX = bigX
        self.bigY = bigY

    def getCenterX(self):
        return int(roundHalfUp((self.bigX + self.smallX) / 2))

    def getCenterY(self):
        return int(roundHalfUp((self.bigY + self.smallY) / 2))

    def getWidth(self):
        width = (self.bigX - self.smallX)
        # Not really considered a box if it is out of bounds
        return (width if width > 0 else 0)

    def getHeight(self):
        height = (self.bigY - self.smallY)
        # Not really considered a box if it is out of bounds
        return (height if height > 0 else 0)

    def getArea(self):
        return self.getWidth() * self.getHeight()

    def isBigEnough(self):
        return True if self.getArea() > Bounding_Box.areaThreshold else False

class KeypointsArray:
    """
        Class which stores keypoints in an array for vectorization purposes
    """
    def __init__(self, xArr, yArr, zArr = None):
        self.x = xArr
        self.xInt = roundHalfUp(xArr).astype(int)
        self._y = yArr
        self.yInt = roundHalfUp(yArr).astype(int)
        self.visibility = np.zeros((Fish.number_of_keypoints))
        self.depth = np.zeros((Fish.number_of_keypoints))
        if zArr is not None:
            self.depth[:zArr.shape[0]] = zArr

        # right now the visibility only depends on if the point is in the frame
        self.visibility = self.inBoundsMask

    @ property
    def inBoundsMask(self):
        isXInBounds = (self.xInt >= 0) * (self.xInt < imageSizeX)
        isYInBounds = (self.yInt >= 0) * (self.yInt < imageSizeY)
        return isXInBounds * isYInBounds
    # y gets modified when getting reflected fishes, hence why it has to be a bit different
    @ property
    def y(self):
        return self._y
    @ y.setter
    def y(self,value):
        #self.keypointsArray[1,:] = value
        self._y = value
        self.yInt = roundHalfUp(value).astype(int)


class Fish:
    """
        Class representing a fish

        Static Properties:
            number_of_keypoints(int):  number of keypoints, set to 12
            number_of_backbone_points: number of backbone_points, set to 10

            proj_params (string): path to mat file containing projection parameters
            lut_b_tail = path to mat file containing the look up table for bottom view
            lut_s_tail = path to mat file containing the look up table for side view
            fish_shapes = path to mat file containing the look up table for fish positions
            lut_b_tail_mat (dict): dictionary of loaded lut_b
            lut_b_tail (numpy array): numpy array of lut_b
            lut_s_tail_mat (dict): dictionary of loaded lut_s
            lut_s_tail numpy array): numpy array of lut_s

        Properties:
            x (numpy array): 22 parameter vector representing the fish position
            fishlen (float): fish length

            graysContainer (list): containing the grayscale images from the three camera views: 'B', 'S1', 'S2'
            depthsContainer (list): containing depth for each image
            annotationsType (AnnotationsType): type depicting which annotations are desired

            # For keypoint annotations, default
            keypointsListContainer (list): Container for the keypoints as seen from each camera view
            boundingBoxContainer (list): Container for the bounding boxes as seen from each camera view

            # For segmentation annotations
            contourContainer (list): Container for the contours of the fish as seen from each camera view

            # TODO: check if these should be deleted
            cropsContainer (list): Container for the crops of the images from each camera view
            coor_3d (numpy array): coordinates in 3d space

        Methods:
            __init__, args(float, numpy array, AnnotationsType = keypoint): creates an instance of a fish as depicted by
                its 22 parameter vector an length.  It also sets the fish instance with the annotations you want, by
                default it is set to keypoint annotations.

            draw(): generates the images of the fish as seen from all three camera views.  It assigns values to the
                graysContainer,  depthsContainer, keypointsListContainer/contourContainer, and boundingBoxContainer.
    """
    # Static properties of Fish Class
    number_of_keypoints = 12
    number_of_backbone_points = 10
    proj_params = 'proj_params_101019_corrected_new'
    proj_params = inputsFolder + proj_params
    mat = loadmat(proj_params)
    proj_params = mat['proj_params']

    lut_b_tail = 'lut_b_tail.mat'
    lut_s_tail = 'lut_s_tail.mat'
    fish_shapes = 'generated_pose_100_percent.mat'
    fish_shapes = inputsFolder + fish_shapes

    lut_s_tail_mat = loadmat(inputsFolder + lut_b_tail)
    lut_b_tail = lut_s_tail_mat['lut_b_tail']

    lut_s_tail_mat = loadmat(inputsFolder + lut_s_tail)
    lut_s_tail = lut_s_tail_mat['lut_s_tail']

    def __init__(self, fishlen, x, annotationsType = AnnotationsType.keypoint):
        self.x = x
        self.fishlen = fishlen

        self.graysContainer = []
        self.depthsContainer = []
        self.annotationsType = annotationsType

        # For keypoint annotations, default
        self.keypointsListContainer = [[], [], []]
        # New version of the above
        self.keypointContainer = []
        self.boundingBoxContainer = [Bounding_Box(),
                                     Bounding_Box(),
                                     Bounding_Box()]

        # For segmentation annotations
        self.contourContainer = [None, None, None]

        # Old version of the script used these
        self.cropsContainer = []
        self.coor_3d = None

    def draw(self):
        [gray_b, gray_s1, gray_s2, crop_b, crop_s1, crop_s2, c_b, c_s1, c_s2, eye_b, eye_s1, eye_s2, self.coor_3d] = \
            return_graymodels_fish(self.x, Fish.proj_params, self.fishlen, imageSizeX, imageSizeY)



        self.graysContainer = [gray_b, gray_s1, gray_s2]
        self.cropsContainer = [crop_b, crop_s1, crop_s2]

        if self.annotationsType == AnnotationsType.segmentation:
            for viewIdx in range(3):
                ########################
                gray = self.graysContainer[viewIdx]
                gray = gray.astype(np.uint8)
                contours, _ = cv.findContours(gray, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)
                numberOfContours = len(contours)
                if numberOfContours != 0:
                    contours = np.squeeze(contours[0])
                    if contours.ndim != 1:
                        self.contourContainer[viewIdx] = contours


        coorsContainer = [c_b, c_s1, c_s2]
        eyesContainer = [eye_b, eye_s1, eye_s2]
        depthsContainer = [imageSizeY - c_s1[1, :], c_b[0, :], c_b[1, :]]
        for viewIdx in range(3):
            coors = coorsContainer[viewIdx]
            eyes = eyesContainer[viewIdx]
            depths = depthsContainer[viewIdx]

            x = np.concatenate((coors[0,:], eyes[0,:]))
            y = np.concatenate((coors[1,:], eyes[1,:]))
            keypointsArray = KeypointsArray(x,y,depths)
            self.keypointContainer.append(keypointsArray)

        # Creating Depth Arrays, Updating Eyes, and getting their bounding boxes
        for viewIdx in range(3):
            gray = self.graysContainer[viewIdx]
            # keypointsList = self.keypointsListContainer[viewIdx]
            keypointsArray = self.keypointContainer[viewIdx]
            #  Creating Depth Arrays  img,  y coor,   x coor,   depth coor
            xArr = keypointsArray.x[:Fish.number_of_backbone_points]
            yArr = keypointsArray.y[:Fish.number_of_backbone_points]
            depth = keypointsArray.depth[:Fish.number_of_backbone_points]
            depthIm = createDepthArr(gray, yArr, xArr, depth)
            (self.depthsContainer).append(depthIm)

            # Updating the depth to be able to compare it after merging the images
            mask4Points = keypointsArray.inBoundsMask
            keypointsArray.depth[mask4Points] = \
                depthIm[keypointsArray.yInt[mask4Points], keypointsArray.xInt[mask4Points]]

            self.keypointContainer[viewIdx] = keypointsArray

            # Finding the bounding box
            nonZeroPoints = np.argwhere(gray != 0)
            if len(nonZeroPoints) != 0:
                [smallY, smallX, bigY, bigX] = [np.min(nonZeroPoints[:, 0]), np.min(nonZeroPoints[:, 1]),
                                                np.max(nonZeroPoints[:, 0]), np.max(nonZeroPoints[:, 1])]

                boundingBox = Bounding_Box(smallX, smallY, bigX, bigY)
                self.boundingBoxContainer[viewIdx] = boundingBox

class Aquarium:
    """
        Class representing a container for the fishes.  It serves to merge all the images of the fishes aswell as get
        the annotations of the fishes.

        Static Properties:
            maxAmountOfFishInAllViews (int): the maximum amount of fishes that can be visible in all views.
            patchy_noise (boolean): True or False representing whether or not to add patchy noise
            amountOfData (int): Amount of data that will be generated, used to split the training data set
            fractionToBe4Training (float): Fraction representing the amount of data that should be used for training
            biggestIdx4TrainingData (float): amountOfData * fractionToBe4Training, the biggest Idx that will be used
                to for Training data

            imagesSubFolder (string): sub path to the images folder
            labelsSubFolder (string): sub path to the labels folder

        Properties:
            idx (int): number representing the index of which data file is being generated
                    self.dataFolderPath = dataFolderPath
            dataFolderPath (string): path to the parent folder in which the data will be saved to
            fishList (list): list that will contain fish objects
            allGContainer (list): list that will contain all the grayscale images
            finalGraysContainer (list): container with the merged images, len of 3 for the three camera views
            allGDepthContainer (list): list that will contain the depth of the grayscale images
            finalDepthsContainer (list): container with the merged depth, len of 3 for the three camera views,
                usefull for updating the visibility of the fish keypoints
            amountOfFish (int): variable for the amount of fishes
            reflectedFishContainer (list): of size three for consistency, but only the 2nd and 3rd element are really
                used.  Need to keep track of the annotations.
            annotationsType (AnnotationsType): flag used to tell which type of annotations are needed

        Methods:
            __init__, args(int, AnnotationsType, dataFolderPath): Creates an instance of the aquarium class.  It
                initializes a lot of the variables and calls generateRandomConfiguration

            generateRandomConfiguration() -> None: Creates a random set of fishes, more info on the actual method
            draw() -> None: Computes the images of each of the three views resulting from the set of fishes
            addReflections() -> None: Add instances of reflected fishes if they are past the water level
            updateFishVisibility() -> None: Goes through each of the keypoint list and updates if the points ended up
                visible or not
            addNoise() -> None: adds static noise to each of the final images of the fishes
            addPatchyNoise() -> None: randomly draws noisy disks on the images to simulate fog
            getGrays() -> list: returns list containing the final images of the fishes
            saveGrays() -> None: saves the three images of the fishes as a rgb png
            createSegmentationAnnotations() -> None: creates segmentation annotations according to the YOLO format
            createKeypointAnnotations() -> None:  creates keypoints annotations according to the YOLO format
            saveAnnotations() -> None: calls createKeypointAnnotations or createSegmentationAnnotations
                depending on the annotations type
    """

    # Aquarium Class Static Properties
    maxAmountOfFishInAllViews = 7
    patchy_noise = True
    amountOfData = 50000
    fractionToBe4Training = .9
    biggestIdx4TrainingData = amountOfData * fractionToBe4Training

    imagesSubFolder = 'images/'
    labelsSubFolder = 'labels/'

    aquariumVariables = ['fishInAllViews','fishInB', 'fishInS1', 'fishInS2', 'fishInEdges','overlapping']
    fishVectListKey = 'fishVectList'

    def __init__(self, idx,annotationsType = AnnotationsType.keypoint ,dataFolderPath = 'data/',**kwargs):

        self.idx = idx
        self.dataFolderPath = dataFolderPath

        self.fishList = []
        self.allGContainer = []
        self.finalGraysContainer = []
        self.allGDepthContainer = []
        self.finalDepthsContainer = []
        self.amountOfFish = 0
        self.waterLevel = None
        self.reflectedFishContainer = [[], [], []]
        self.annotationsType = annotationsType


        # Detecting which type of aquarium you want to generate
        aquariumVariablesDict = {'fishInAllViews':0, 'fishInB':0, 'fishInS1':0, 'fishInS2':0, 'fishInEdges':0,
                                 'overlapping':0}
        wasAnAquariumVariableDetected = False
        wasAnAquariumPassed = False
        for key in kwargs:
            if key in Aquarium.aquariumVariables:
                aquariumVariablesDict[key] = kwargs.get(key)
                wasAnAquariumVariableDetected = True
            if key is Aquarium.fishVectListKey:
                wasAnAquariumPassed = True
                fishVectList = kwargs.get(key)
            if key == 'waterLevel':
                self.waterLevel = kwargs.get(key)


        # Initializing the aquarium based on the args
        if not wasAnAquariumPassed:
            if wasAnAquariumVariableDetected:
                fishVectList = self.generateFishListGivenVariables(aquariumVariablesDict)
            else:
                fishVectList = self.generateRandomConfiguration()
        self.amountOfFish = len(fishVectList)


        # Create the list of fishes
        for fishVect in fishVectList:
            fishlen = fishVect[0]
            x = fishVect[1:]
            fish = Fish(fishlen, x, self.annotationsType)
            (self.fishList).append(fish)

        if self.waterLevel is None:
            # Set it to a random value
            self.waterLevel = 0
            shouldThereBeWater = True if np.random.rand() > .5 else False
            if shouldThereBeWater:
                self.waterLevel = np.random.randint(10, high=60)


    def generateFishListGivenVariables(self,aquariumVariablesDict):
        fishInAllViews = aquariumVariablesDict.get('fishInAllViews')
        fishInB = aquariumVariablesDict.get('fishInB')
        fishInS1 = aquariumVariablesDict.get('fishInS1')
        fishInS2 = aquariumVariablesDict.get('fishInS2')
        overlapping = aquariumVariablesDict.get('overlapping')
        fishInEdges = aquariumVariablesDict.get('fishInEdges')
        fishList = generateAquariumOverlapsAndWithoutBoundsAndEdges(fishInAllViews, fishInB, fishInS1,
                                                                    fishInS2, overlapping, fishInEdges )
        return fishList


    def generateRandomConfiguration(self):
        self.fishInAllView = np.random.randint(0, Aquarium.maxAmountOfFishInAllViews)

        self.overlapping = 0
        for jj in range(self.fishInAllView):
            shouldItOverlap = np.random.rand()
            if shouldItOverlap > .5:
                self.overlapping += 1

        self.fishesInB = np.random.poisson(2)
        self.fishesInS1 = np.random.poisson(2)
        self.fishesInS2 = np.random.poisson(2)

        self.fishesInEdges = 0
        if np.random.rand() > .5:
            self.fishesInEdges = np.random.poisson(3)


        self.amountOfFish = self.fishInAllView + self.overlapping + self.fishesInB \
                            + self.fishesInS1 + self.fishesInS2 + self.fishesInEdges

        self.s1FishReflected = []
        self.s2FishReflected = []

        aquarium = generateAquariumOverlapsAndWithoutBoundsAndEdges(self.fishInAllView, self.fishesInB, self.fishesInS1,
                                                                    self.fishesInS2, self.overlapping, self.fishesInEdges)

        return aquarium


    def draw(self):
        # TODO replacing these by adding another dimension to a numpy array might make it better
        for viewIdx in range(3):
            allG = np.zeros((self.amountOfFish, imageSizeY, imageSizeX ))
            allGDepth = np.zeros((self.amountOfFish, imageSizeY, imageSizeX))
            self.allGContainer.append(allG)
            self.allGDepthContainer.append( allGDepth)

        # Getting the grayscale images as well as depth images from the fishes
        for fishIdx, fish in enumerate(self.fishList):
            fish.draw()
            graysContainer = fish.graysContainer
            depthsContainer = fish.depthsContainer

            for viewIdx in range(3):
                gray = graysContainer[viewIdx]
                depth = depthsContainer[viewIdx]
                gContainer = self.allGContainer[viewIdx]
                depthContainer = self.allGDepthContainer[viewIdx]

                gContainer[fishIdx,...] = gray
                depthContainer[fishIdx, ...] = depth

                # Updating
                self.allGContainer[viewIdx] = gContainer
                self.allGDepthContainer[viewIdx] = depthContainer

        if self.waterLevel > 0 :
            self.addReflections()

        # Merging all the images together
        for viewIdx in range(3):
            allG = self.allGContainer[viewIdx]
            allGDepth = self.allGDepthContainer[viewIdx]

            finalGray, finalDepth = mergeGreys(allG, allGDepth)
            self.finalGraysContainer.append(finalGray)
            self.finalDepthsContainer.append(finalDepth)
        # After merging all the pictures some parts of the fish may have become hidden
        self.updateFishVisibility()

        self.addNoise()
        if Aquarium.patchy_noise :
            self.addPatchyNoise()


    def addReflections(self):
        # Only viewS1 and viewS2
        for viewIdx in range(1,3):
            reflectedFishList = self.reflectedFishContainer[viewIdx]
            allG = self.allGContainer[viewIdx]
            allGDepth = self.allGDepthContainer[viewIdx]

            for fishIdx, fish in enumerate(self.fishList):
                boundingBox = fish.boundingBoxContainer[viewIdx]
                topPoint = int(boundingBox.smallY)

                if topPoint <= self.waterLevel and not(topPoint <= 0):

                    # The fish will have a reflection
                    # TODO update this so that you also consider when the fish falls within the range of the reflection
                    gray = fish.graysContainer[viewIdx]
                    depth = fish.depthsContainer[viewIdx]
                    # Adding the reflections to the images
                    grayFlipped = np.flipud(np.copy(gray))
                    grayDepthFlipped = np.flipud(depth)

                    gray[0:topPoint, :] = np.copy(1 * grayFlipped[-topPoint * 2:-topPoint, :])
                    depth[0:topPoint, :] = grayDepthFlipped[-topPoint * 2:-topPoint, :]
                    # Adding a reflected fish object for annotations
                    reflectedFish = copy.deepcopy(fish)
                    reflectedKeypointsArr = reflectedFish.keypointContainer[viewIdx]

                    # the fish was flipped, so we need to flip the y coordinates of the keypoints
                    y = reflectedKeypointsArr.y
                    y = (imageSizeY - 1) - y
                    y -= imageSizeY - (2 * topPoint)
                    reflectedKeypointsArr.y = y

                    mask = reflectedKeypointsArr.inBoundsMask
                    # updating the depth so that we can keep track of it
                    reflectedKeypointsArr.depth[mask] = \
                        depth[ reflectedKeypointsArr.yInt[mask], reflectedKeypointsArr.xInt[mask] ]
                    # When reflecting the fish its possible that a keypoint got cut off
                    reflectedKeypointsArr.visibility = mask


                    reflectedBoundingBox = reflectedFish.boundingBoxContainer[viewIdx]
                    boundingBoxYs = np.array([reflectedBoundingBox.smallY, reflectedBoundingBox.bigY])
                    boundingBoxYs = (imageSizeY - 1) - boundingBoxYs
                    boundingBoxYs -= imageSizeY - (2 * topPoint)
                    [reflectedBoundingBox.bigY, reflectedBoundingBox.smallY] = boundingBoxYs
                    reflectedBoundingBox.smallY = np.clip(reflectedBoundingBox.smallY, 0, imageSizeY - 1)
                    reflectedBoundingBox.bigY = np.clip(reflectedBoundingBox.bigY, 0, imageSizeY - 1)

                    # Cutting off the top of the reflection to better simulate
                    # the observed reflections
                    probToCutOffReflection = .5
                    shouldBoxBeCutOff = True if np.random.rand() < probToCutOffReflection else False
                    if shouldBoxBeCutOff:
                        heightOfReflectionsBoundingBox = reflectedBoundingBox.bigY - reflectedBoundingBox.smallY
                        amountToChopOff = np.random.randint(0, heightOfReflectionsBoundingBox + 1)

                        # the small values represents the higher rows in the array
                        gray[reflectedBoundingBox.smallY : reflectedBoundingBox.smallY + amountToChopOff, :] = 0
                        depth[reflectedBoundingBox.smallY: reflectedBoundingBox.smallY + amountToChopOff, :] = 0

                        reflectedBoundingBox.smallY = \
                        reflectedBoundingBox.bigY - (heightOfReflectionsBoundingBox - amountToChopOff)
                        # Setting the y values to be within the bounding box
                        reflectedKeypointsArr.y[reflectedKeypointsArr.y < reflectedBoundingBox.smallY] = \
                            reflectedBoundingBox.smallY

                        # setting it to nan since everything that was chopped is no longer there
                        # will be set to invisible when updatevisibility is called
                        reflectedKeypointsArr.depth[ reflectedKeypointsArr.y < reflectedBoundingBox.smallY ] = np.nan
                    # Updating the containers
                    reflectedFish.boundingBoxContainer[viewIdx] = reflectedBoundingBox
                    reflectedFish.keypointContainer[viewIdx] = reflectedKeypointsArr

                    # In case you want segmentation annotations
                    if self.annotationsType == AnnotationsType.segmentation:
                        tempMask = np.zeros((imageSizeY, imageSizeX))
                        tempMask[0:topPoint, :] = np.copy(1 * grayFlipped[-topPoint * 2:-topPoint, :])
                        if shouldBoxBeCutOff:
                            tempMask[reflectedBoundingBox.smallY - amountToChopOff: reflectedBoundingBox.smallY, :] = 0
                        tempMask = tempMask.astype(np.uint8)

                        contours, _ = cv.findContours(tempMask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)
                        numberOfContours = len(contours)
                        if numberOfContours != 0:
                            contours = np.squeeze(contours[0])
                            if contours.ndim != 1:
                                reflectedFish.contourContainer[viewIdx] = contours

                    allG[fishIdx,...] = gray
                    allGDepth[fishIdx, ...] = depth
                    reflectedFishList.append(reflectedFish)

            # Updating
            self.reflectedFishContainer[viewIdx] = reflectedFishList
            self.allGContainer[viewIdx] = allG
            self.allGDepthContainer[viewIdx] = allGDepth

    def updateFishVisibility(self):

        for viewIdx in range(3):
            finalG = self.finalGraysContainer[viewIdx]
            finalGDepth = self.finalDepthsContainer[viewIdx]
            # Real fishes
            for fishIdx, fish in enumerate(self.fishList):
                keypointsArray = fish.keypointContainer[viewIdx]
                mask4Points = keypointsArray.inBoundsMask
                arePointStillVis = \
                    (keypointsArray.depth[mask4Points] ==
                     finalGDepth[keypointsArray.yInt[mask4Points], keypointsArray.xInt[mask4Points]])
                keypointsArray.visibility[mask4Points] = arePointStillVis
                # Updating the containers/Lists
                fish.keypointContainer[viewIdx] = keypointsArray
                self.fishList[fishIdx] = fish

            # Fish reflections
            reflectedFishList = self.reflectedFishContainer[viewIdx]
            for fishIdx, fish in enumerate(reflectedFishList):
                keypointsArray = fish.keypointContainer[viewIdx]
                mask = keypointsArray.inBoundsMask
                arePointStillVis = (keypointsArray.depth[mask] ==
                                    finalGDepth[ keypointsArray.yInt[mask], keypointsArray.xInt[mask] ])
                keypointsArray.visibility[mask] = arePointStillVis
                # Updating
                fish.keypointContainer[viewIdx] = keypointsArray
                reflectedFishList[fishIdx] = fish
            self.reflectedFishContainer[viewIdx] = reflectedFishList


    def addNoise(self):
        filter_size = 2 * roundHalfUp(np.random.rand(3)) + 3
        sigma = np.random.rand(3) + 0.5
        for viewIdx in range(3):
            gray = self.finalGraysContainer[viewIdx]
            kernel = cv.getGaussianKernel(int(filter_size[viewIdx]), sigma[viewIdx])
            gray = cv.filter2D(gray, -1, kernel)
            gray = uint8(gray)

            # Converting to range [0,1]
            # TODO Change this part so it is not modifying the grayscale image
            # can make it more elegant
            maxGray = max(gray.flatten())
            if maxGray != 0:
                gray = gray / max(gray.flatten())
            else:
                gray[0, 0] = 1
            gray = imGaussNoise(gray, (np.random.rand() * np.random.normal(50, 10)) / 255,
                                (np.random.rand() * 50 + 20) / 255 ** 2)
            # Converting Back
            if maxGray != 0:
                gray = gray * (255 / max(gray.flatten()))
            else:
                gray[0, 0] = 0
                gray = gray * 255
            gray = uint8(gray)

            # Updating
            self.finalGraysContainer[viewIdx] = gray

    def addPatchyNoise(self):
        fishList = self.fishList
        for viewIdx in range(3):
            gray = self.finalGraysContainer[viewIdx]

            boundingBoxList = [ fish.boundingBoxContainer[viewIdx] for fish in fishList]

            pvar = np.random.poisson(0.2)
            if (pvar > 0):
                for i in range(1, int(np.floor(pvar + 1))):
                    # No really necessary, but just to ensure we do not lose too many
                    # patches to fishes barely visible or fishes that do not appear in the view
                    idxListOfPatchebleFishes = [idx for idx , boundingBox in enumerate(boundingBoxList) if boundingBox.isBigEnough]
                    amountOfPossibleCenters = len(idxListOfPatchebleFishes)

                    finalVar_mat = np.zeros((imageSizeY, imageSizeX))
                    amountOfCenters = np.random.randint(0, high=(amountOfPossibleCenters + 1))
                    for centerIdx in range(amountOfCenters):
                        # y, x
                        center = np.zeros((2))
                        shouldItGoOnAFish = True if np.random.rand() >.5 else False
                        if shouldItGoOnAFish:
                            boundingBox = boundingBoxList[ idxListOfPatchebleFishes[centerIdx] ]
                            center[0] = (boundingBox.getHeight() * (np.random.rand() - .5)) + boundingBox.getCenterY()
                            center[1] = (boundingBox.getWidth() * (np.random.rand() - .5 )) + boundingBox.getCenterX()
                            center = center.astype(int)
                            # clip just in case we went slightly out of bounds
                            center[0] = np.clip(center[0], 0, imageSizeY - 1)
                            center[1] = np.clip(center[1], 0, imageSizeX - 1)

                        else:
                            center[0] = np.random.randint(0, high=imageSizeY)
                            center[1] = np.random.randint(0, high=imageSizeX)

                        zeros_mat = np.zeros((imageSizeY, imageSizeX))
                        zeros_mat[int(center[0]) - 1, int(center[1]) - 1] = 1
                        randi = (2 * np.random.randint(5, high=35)) + 1
                        se = cv.getStructuringElement(cv.MORPH_ELLIPSE, (randi, randi))
                        zeros_mat = cv.dilate(zeros_mat, se)
                        finalVar_mat += zeros_mat

                    maxG = max(gray.flatten())
                    gray = gray / maxG
                    # gray_b = imnoise(gray_b, 'localvar', var_mat * 3 * (np.random.rand() * 60 + 20) / 255 ** 2)
                    gray = random_noise(gray, mode='localvar', local_vars=(finalVar_mat * 3 * (
                            np.random.rand() * 60 + 20) / 255 ** 2) + .00000000000000001)
                    gray = gray * (maxG / max(gray.flatten()))

            # Updating
            self.finalGraysContainer[viewIdx] = gray

    def getGrays(self):
        return self.finalGraysContainer

    def saveGrays(self):
        rgb = np.zeros((imageSizeY,imageSizeX,3))
        for channel in range(3): rgb[..., channel] = self.finalGraysContainer[channel]
        subFolder = 'train/' if self.idx < Aquarium.biggestIdx4TrainingData else 'val/'
        imagePath = self.dataFolderPath + Aquarium.imagesSubFolder + subFolder
        strIdxInFormat = format(self.idx, '06d')
        filename = 'zebrafish_1_' + strIdxInFormat + '.png'
        imagePath += filename
        cv.imwrite(imagePath, rgb)

    #For debugging purposes
    def saveCatGrays(self):
        subFolder = 'train/' if self.idx < Aquarium.biggestIdx4TrainingData else 'val/'
        imagePath = self.dataFolderPath + Aquarium.imagesSubFolder + subFolder
        strIdxInFormat = format(self.idx, '06d')
        filename = 'zebrafish_1_' + strIdxInFormat + '.png'
        imagePath += filename
        catGrays = np.concatenate([gray for gray in self.finalGraysContainer], axis=0)
        cv.imwrite(imagePath,catGrays)

    def createSegmentationAnnotations(self):
        subFolder = 'train/' if self.idx < Aquarium.biggestIdx4TrainingData else 'val/'
        labelsPath = self.dataFolderPath + Aquarium.labelsSubFolder + subFolder
        strIdxInFormat = format(self.idx, '06d')
        filename = 'zebrafish_1_' + strIdxInFormat + '.txt'
        labelsPath += filename

        f = open(labelsPath, 'w')
        for viewIdx in range(3):
            for fish in self.fishList:
                contour = fish.contourContainer[viewIdx]
                boundingBox = fish.boundingBoxContainer[viewIdx]

                # Should add a method to the bounding box, boundingBox.isSmallFishOnEdge()
                if (contour is not None) and boundingBox.isBigEnough():
                    f.write(str(viewIdx) + ' ')

                    for pIdx in range(contour.shape[0]):
                        y = contour[pIdx, 1]
                        x = contour[pIdx, 0]

                        f.write(str(x / imageSizeX) + ' ')
                        f.write(str(y / imageSizeY) + ' ')

                    f.write('\n')

            reflectedFishList = self.reflectedFishContainer[viewIdx]
            for fish in reflectedFishList:
                contour = fish.contourContainer[viewIdx]
                boundingBox = fish.boundingBoxContainer[viewIdx]

                if (contour is not None) and boundingBox.isBigEnough():
                    f.write(str(viewIdx) + ' ')

                    for pIdx in range(contour.shape[0]):
                        y = contour[pIdx, 1]
                        x = contour[pIdx, 0]

                        f.write(str(x / imageSizeX) + ' ')
                        f.write(str(y / imageSizeY) + ' ')

                    f.write('\n')

    def createKeypointAnnotations(self):
        subFolder = 'train/' if self.idx < Aquarium.biggestIdx4TrainingData else 'val/'
        labelsPath = self.dataFolderPath + Aquarium.labelsSubFolder + subFolder
        strIdxInFormat = format(self.idx, '06d')
        filename = 'zebrafish_1_' + strIdxInFormat + '.txt'
        labelsPath += filename

        f = open(labelsPath, 'w')
        for viewIdx in range(3):
            for fish in self.fishList:
                keypointsArray = fish.keypointContainer[viewIdx]
                boundingBox = fish.boundingBoxContainer[viewIdx]

                # Should add a method to the bounding box, boundingBox.isSmallFishOnEdge()
                if boundingBox.getArea() != 0:
                    f.write(str(viewIdx) + ' ')
                    f.write(str(boundingBox.getCenterX()/imageSizeX) + ' ' + str(boundingBox.getCenterY()/imageSizeY) + ' ')
                    f.write(str(boundingBox.getWidth()/imageSizeX) + ' ' + str(boundingBox.getHeight()/imageSizeY) + ' ')

                    xArr = keypointsArray.x
                    yArr = keypointsArray.y
                    vis = keypointsArray.visibility
                    for pointIdx in range(Fish.number_of_keypoints):
                        # Visibility is set to zero if they are out of bounds
                        # Just got to clip them so that YOLO does not throw an error
                        x = np.clip(xArr[pointIdx], 0, imageSizeX - 1)
                        y = np.clip(yArr[pointIdx], 0, imageSizeY - 1)
                        f.write(str(x / imageSizeX) + ' ' + str(y / imageSizeY)
                                + ' ' + str(int(vis[pointIdx])) + ' ')
                    f.write('\n')

            # Writing the reflections
            reflectedFishList = self.reflectedFishContainer[viewIdx]
            for fish in reflectedFishList:
                keypointsArray = fish.keypointContainer[viewIdx]
                boundingBox = fish.boundingBoxContainer[viewIdx]

                if boundingBox.getArea() != 0:
                    f.write(str(viewIdx) + ' ')
                    f.write(str(boundingBox.getCenterX()/imageSizeX) + ' ' + str(boundingBox.getCenterY()/imageSizeY) + ' ')
                    f.write(str(boundingBox.getWidth()/imageSizeX) + ' ' + str(boundingBox.getHeight()/imageSizeY) + ' ')

                    xArr = keypointsArray.x
                    yArr = keypointsArray.y
                    vis = keypointsArray.visibility
                    for pointIdx in range(Fish.number_of_keypoints):
                        # Visibility is set to zero if they are out of bounds
                        # Just got to clip them so that YOLO does not throw an error
                        x = np.clip(xArr[pointIdx], 0, imageSizeX - 1)
                        y = np.clip(yArr[pointIdx], 0, imageSizeY - 1)
                        f.write(str(x / imageSizeX) + ' ' + str(y / imageSizeY)
                                + ' ' + str(int( vis[pointIdx] )) + ' ')
                    f.write('\n')

    def saveAnnotations(self):
        if self.annotationsType == AnnotationsType.keypoint:
            self.createKeypointAnnotations()

        elif self.annotationsType == AnnotationsType.segmentation:
            self.createSegmentationAnnotations()