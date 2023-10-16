import numpy as np
import imageio
import cv2 as cv
# import programs.programsForGeneratingRandomFishes as programsFile

imageSizeX = 648
imageSizeY = 488

dataParentFolder = '../data/'
imagesFolder = 'images'
labelsFolder = 'labels'

imageFileName = '/train/zebrafish_1_000020.png'
absolutePathToImage = dataParentFolder + imagesFolder + imageFileName
parentPath = absolutePathToImage[:-35]
subPath = absolutePathToImage[-29:]
name = subPath[-22:]
name = name[:-4]
absolutePathToLabel = parentPath + 'labels' + subPath[:-4] + '.txt'

def getImgWithAnnotations(absolutePathToImage):
    parentPath = absolutePathToImage[:-35]
    subPath = absolutePathToImage[-29:]
    absolutePathToLabel = parentPath + 'labels' + subPath[:-4] + '.txt'
    name = subPath[-22:]
    name = name[:-4]

    im = imageio.imread( absolutePathToImage)
    im = np.array(im)
    # The channels get flipped for some reason
    viewsContainer = [im[..., 2 - idx] for idx in range(3)]
    imageHeight = viewsContainer[0].shape[0]
    imageWidth = viewsContainer[0].shape[1]
    originalImagesCat = np.concatenate([view for view in viewsContainer], axis= 0)

    red = [0, 0, 255]
    green = [0, 255, 0]
    blue = [255, 0, 0]
    keypointsListContainer = [[],[],[]]
    boxListContainer = [[],[],[]]

    # Reading the data and seperating it approprietly
    with open(absolutePathToLabel,'r') as file:

        for line in file:
            keypointsArr = np.zeros((12, 3))
            words = line.split()
            keypoints = words[5:]

            viewIdx = int(words[0])
            keypointsList = keypointsListContainer[viewIdx]
            boxList = boxListContainer[viewIdx]

            for x in range(12):
                xInFormat = x * 3
                keypointsArr[x,:] = [float(keypoints[xInFormat]),float(keypoints[xInFormat + 1]),float(keypoints[xInFormat + 2])]
            keypointsList.append(keypointsArr)

            bBox = words[1:5]
            [x, y, w, h] = [int(np.ceil(float(bBox[0]) * imageWidth)), int(np.ceil(float(bBox[1]) * imageHeight)),
                            int(np.ceil(float(bBox[2]) * imageWidth)), int(np.ceil(float(bBox[3]) * imageHeight))]
            [tx, bx, ty, by] = [x + int(np.ceil(w/2)) ,x - int(np.ceil(w/2)),y + int(np.ceil(h/2)) , y - int(np.ceil(h/2))]
            bBoxConverted = [tx, bx, ty, by]
            boxList.append(bBoxConverted)

    # Turning them into numpy arrays, because the old program did this
    for viewIdx in range(3):
        keypointsList = keypointsListContainer[viewIdx]
        keypointsList = np.array(keypointsList)
        keypointsList[:,:,0] *= imageWidth
        keypointsList[:,:,1] *= imageHeight
        keypointsListContainer[viewIdx] = keypointsList

    # Drawing the annotations
    viewsWithAnnotationsContainer = []
    for viewIdx in range(3):
        keypointsList = keypointsListContainer[viewIdx]
        bBoxList = boxListContainer[viewIdx]
        originalView = viewsContainer[viewIdx]
        rgb = np.zeros((originalView.shape[0], originalView.shape[1],3))
        for channel in range(3): rgb[...,channel] = originalView

        # Drawing the annotated keypoints
        for keypoints in keypointsList:
            for pointIdx, point in enumerate(keypoints):
                [col, row] = np.floor(point[:2]).astype(int)
                vis = point[2]
                if pointIdx < 10:
                    # It is a backbone point
                    if vis:
                        rgb[row, col] = green
                    else:
                        rgb[row, col] = blue
                else:
                    # It is an eye
                    if vis:
                        rgb[row, col] = red
                    else:
                        rgb[row, col] = blue

        # Drawing the Boxes
        for box in bBoxList:
            box[:2] = np.clip(box[:2], 0, imageSizeX - 1)
            box[2:] = np.clip(box[2:], 0, imageSizeY - 1)
            [tx, bx, ty, by] = box
            rgb[ty, bx:tx, :] = red
            rgb[by, bx:tx, :] = red
            rgb[by:ty, bx, :] = red
            rgb[by:ty, tx, :] = red

        viewsWithAnnotationsContainer.append(rgb)

    viewsWithAnnotationsCat = np.concatenate([im for im in viewsWithAnnotationsContainer])
    blueLine = np.zeros((imageHeight * 3, 1,3))
    blueLine[...,0] = 255
    rgb = np.zeros((imageHeight * 3, imageWidth, 3))
    for channel in range(3): rgb[...,channel] = originalImagesCat
    originalImagesCat = rgb
    results = np.concatenate((originalImagesCat,blueLine,viewsWithAnnotationsCat), axis =1)
    return results, name

    cv.imwrite('test_WithOverlay.png', results)

