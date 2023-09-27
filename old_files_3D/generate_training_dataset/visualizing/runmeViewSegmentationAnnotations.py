from old_files_3D.generate_training_dataset.programs.visualizeYOLOSegmentationFunction import getImgWithAnnotations
import numpy as np
import cv2 as cv
import os
import random
import shutil
import imageio

# TODO: optimize and clean this program

# Note currently only works with RGB Images, TODO: make it handle concatenated images

# They get saved to the results folder
maxAmountOfFilesToVisualize = 10

resultsFolder = 'results/'
if not os.path.exists(resultsFolder[:-1]):
    os.makedirs(resultsFolder[:-1])
else:
    # reset it
    shutil.rmtree(resultsFolder[:-1])
    os.makedirs(resultsFolder[:-1])

dataTrainFolderPath = '../data/images/train/'
dataValFolderPath = '../data/images/val/'

listOfFilesInTrain = os.listdir(dataTrainFolderPath)
amountOfTrainFiles = len(listOfFilesInTrain)
listOfFilesInVal = os.listdir(dataValFolderPath)
amountOfValFiles = len(listOfFilesInVal)

# avoid repetions
totalAmountOfFiles = amountOfValFiles + amountOfTrainFiles
if totalAmountOfFiles < maxAmountOfFilesToVisualize:
    for fileName in listOfFilesInTrain:
        absolutePath = dataTrainFolderPath + fileName
        im, name = getImgWithAnnotations(absolutePath)
        cv.imwrite(resultsFolder + name + '.png', im)
    for fileName in listOfFilesInVal:
        absolutePath = dataValFolderPath + fileName
        im, name = getImgWithAnnotations(absolutePath)
        cv.imwrite(resultsFolder + name + '.png', im)
else:
    jointList = listOfFilesInTrain + listOfFilesInVal
    indices = range(totalAmountOfFiles)
    randomIndices = random.sample(indices, maxAmountOfFilesToVisualize)
    for idx in randomIndices:
        randomFileName = jointList[idx]
        parentPath = dataValFolderPath if (randomFileName in listOfFilesInVal) else dataTrainFolderPath
        absolutePath = parentPath + randomFileName
        im, name = getImgWithAnnotations(absolutePath)
        cv.imwrite(resultsFolder + name + '.png', im)
