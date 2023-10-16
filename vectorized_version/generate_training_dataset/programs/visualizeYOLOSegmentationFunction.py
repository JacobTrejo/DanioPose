import cv2 as cv
import imageio
import numpy as np
import os

red = [0, 0, 255]
green = [0, 255, 0]
blue = [255, 0, 0]

# The outline is a numpy array where the first column are the x coordinates
# and the second column are y coordinates
def drawMaskFromOutline(img, outline):
    # Got to rewrite it for cv2's wierd format
    outline = [outline.reshape((-1, 1, 2)).astype(np.int32)]
    img = cv.drawContours(img, outline, -1, (1), -1)
    return img

# The outlines List is a list of outlines in the format described above
def drawMasksFromMultipleOutlines(outlineList, imageSizeY = 488, imageSizeX = 648):
    mask = np.zeros((imageSizeY,imageSizeX))
    for outline in outlineList:
        mask = drawMaskFromOutline(mask, outline)
    return mask

def createMaskFromTextFile(textFilePath):
    #Obtaining the outlines
    outlineListContainer = [[], [], []]
    with open(textFilePath, 'r') as file:
        for line in file:
            words = line.split()
            view = int(words[0])
            points = words[1:]
            amountOfPoints = len(points) // 2
            outline = np.zeros((amountOfPoints, 2))
            for pointIdx in range(amountOfPoints):
                [xN, yN] = points[2 * pointIdx:2 * (pointIdx + 1)]
                point = np.array([648 * float(xN), 488 * float(yN)])
                outline[pointIdx, :] = point

            outlineListContainer[view].append(outline)

    # Creating the mask from the lists
    maskContainer = []
    for viewIdx in range(3):
        outlineList = outlineListContainer[viewIdx]

        mask = drawMasksFromMultipleOutlines(outlineList)
        # maskedImage = mask * viewsContainer[viewIdx]
        maskContainer.append(mask)
    # This is in the shape (3, imageSizeY, imageSizeX)
    # rgb = np.array([ mask for mask in maskContainer])

    # This is the shape (imageSizeY, imageSizeX,3)
    rgb = np.stack([mask for mask in maskContainer], axis=2)

    return rgb

def getImgWithAnnotations(absolutePathToImage):
    parentPath = absolutePathToImage[:-35]
    subPath = absolutePathToImage[-29:]
    absolutePathToLabel = parentPath + 'labels' + subPath[:-4] + '.txt'
    name = subPath[-22:]
    name = name[:-4]

    rgbMask = createMaskFromTextFile(absolutePathToLabel)
    rgbIm = imageio.imread(absolutePathToImage)

    cutOutContainer = []
    for channel in range(3):
        mask = rgbMask[...,2 - channel]
        # Imageio flips the channels
        im = rgbIm[..., channel]
        cutOut = mask * im
        cutOutContainer.append(cutOut)
    # catCutOut = np.concatenate( [cutOut for cutOut in cutOutContainer], axis=0)
    # catIm = np.concatenate( [rgbIm[...,channel] for channel in range(3)], axis=0 )
    catCutOut = np.concatenate( [cutOutContainer[2 -  idx] for idx in range(3)], axis=0)
    catIm = np.concatenate( [rgbIm[...,2 - channel] for channel in range(3)], axis=0 )


    # Turning them in rgb
    rgbCatCutOut = np.stack( [catCutOut for _ in range(3)], axis=2)
    rgbCatIm = np.stack( [catIm for _ in range(3)], axis=2)
    imageHeight = catIm.shape[0]
    blueLine = np.zeros((imageHeight,1,3))
    blueLine[...,0] = 255

    result = np.concatenate( (rgbCatIm, blueLine, rgbCatCutOut), axis=1)
    return result, name
