import os
from programs.Aquarium import *
import multiprocessing
from multiprocessing import  Pool,cpu_count
import time
import shutil


# Creating the Dataset Folder for YOLO training dataset
# Suppose to look something like this:
#    - data   -> images -> train
#                 |     -> val
#         |  ->  labels -> train
#                 |     -> val

homepath = 'data/'

if not os.path.exists(homepath[:-1]):
    os.makedirs(homepath[:-1])
else:
    # reset it
    shutil.rmtree(homepath)
    os.makedirs(homepath[:-1])

folders = ['images','labels']
subFolders = ['train','val']
for folder in folders:
    subPath = homepath + folder
    if not os.path.exists(subPath):
        os.makedirs(subPath)
    for subFolder in subFolders:
        subSubPath = subPath + '/' + subFolder
        if not os.path.exists(subSubPath):
            os.makedirs(subSubPath)

def init_pool_process():
    np.random.seed()

def genData(idx):
    aquarium = Aquarium( idx, annotationsType= AnnotationsType.keypoint )
    aquarium.draw()
    # aquarium.saveCatGrays()
    aquarium.saveGrays()
    aquarium.saveAnnotations()

# if __name__ == '__main__':
#     # multiprocessing case
#     print('Process Starting')
#     startTime = time.time()
#     amount = 10
#     pool_obj = multiprocessing.Pool(initializer=init_pool_process)
#     pool_obj.map(genData, range(0,amount))
#     pool_obj.close()
#     endTime = time.time()
#
#     print('Finish Running')
#     print('Average Time: ' + str((endTime - startTime)/amount))


debugging = True
if debugging:
    # Notes: waterLevel variable can be used by any of the cases, defaults to a random value capped at 60 pixels
    #        annotationsType is optional, defaults to keypoint

    # Case where you specify your fish
    # Note: first element is the fish length the rest is the 22 parameter vector x
    fish = [3.8993589130855133, 5.61911477e+00 - 5, 4.22782300e+00 - 18, 7.73263985e+01 + 7, 2.53216701e+00 + np.pi, 1.07559189e-01,
            2.08386998e-02, 4.60060940e-01, -3.09544318e-01, 3.89490747e-01, 4.23302416e-01, 8.78224886e-01,
            5.83942838e-01,
            2.22742787e-02, -1.18175973e-01, 1.17280792e-02, -3.25415602e-01, -.349, .2896, -.17941, -.06075, .08510,
            .65856]
    fish2 = [3.8993589130855133, 5.61911477e+00 - 5, 4.22782300e+00 - 8, 7.73263985e+01 - 7, 0 , 0,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    fishVectList = [fish,fish2]
    aquarium = Aquarium(20 ,fishVectList = fishVectList,annotationsType = AnnotationsType.keypoint )
    # aquarium = Aquarium(20 ,annotationsType = AnnotationsType.segmentation )

    # Case where you specify variables
    # Notes: overlapping and fishInEdges are not constrained to show up in only one view
    # Also overlapping currently overlaps fishes that are in all views and so
    # fishInAllViews >= overlapping
    # aquarium = Aquarium(20, fishInAllViews = 2, overlapping = 1, fishInB = 0, fishInS1 = 0,
    #                     fishInS2 = 0, fishInEdges = 0, waterLevel = 25)

    # # Case where you want a random aquarium
    # aquarium = Aquarium(20)

    aquarium.draw()
    # TO save the images concatenated vertically use:
    # aquarium.saveCatGrays()
    # To save the images in RGB Format use:
    aquarium.saveGrays()
    aquarium.saveAnnotations()


