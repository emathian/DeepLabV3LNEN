from PIL import Image
from tqdm import tqdm
import numpy as np
import cv2
import os, shutil

global TunorMat
global NoTumorMat
global BackgroundMat
global palette
# palette (color map) describes the (R, G, B): Label pair
TumorMat = np.ones((512,512,3), dtype=np.uint8) 
TumorMat[:,:,0] = 0
TumorMat[:,:,1] = 255
TumorMat[:,:,2] = 255


NoTumorMat = np.ones((512,512,3), dtype=np.uint8) 
NoTumorMat[:,:,0] = 192
NoTumorMat[:,:,1] = 192
NoTumorMat[:,:,2] = 192


BackgroundMat = np.ones((512,512,3), dtype=np.uint8) 
BackgroundMat[:,:,0] = 255
BackgroundMat[:,:,1] = 255
BackgroundMat[:,:,2] = 255


palette = { (195,   5,   2) : 0 , # Tumor
         (150,  200, 150) : 1, # No Tumor
         (252,253,252) : 2 # Background
          }
def TumorNoTumorBackgroundIndex(im):
    masknorm = np.zeros((im.shape))
    dT = np.abs(np.sum(im.astype('int8')  - TumorMat.astype('int8') , axis = 2))
    dNT = np.abs(np.sum(im.astype('int8')  - NoTumorMat.astype('int8') , axis = 2))
    dB = np.abs(np.sum(im.astype('int8')  - BackgroundMat.astype('int8') , axis = 2))
    diff_mat = np.zeros((im.shape))
    diff_mat[:,:,0] = dT
    diff_mat[:,:,1] = dNT
    diff_mat[:,:,2] = dB
    arr2dINDEX = np.argmin(diff_mat, axis = 2)

    masknormOri = np.ones((im.shape))
    masknormT = np.multiply(np.multiply(np.transpose(masknormOri, axes=(2,0,1)) ,
                             (np.array(arr2dINDEX == 0)*1)),np.transpose(TumorMat,axes=(2,0,1)))
    masknormNT =  np.multiply(np.multiply( np.transpose(masknormOri, axes=(2,0,1)), 
                             (np.array(arr2dINDEX == 1)*1)), np.transpose(NoTumorMat,axes=(2,0,1)))

    masknormB =  np.multiply(np.multiply(np.transpose(masknormOri, axes=(2,0,1)), 
                             (np.array(arr2dINDEX == 2)*1)), np.transpose(BackgroundMat, axes=(2,0,1)))
    masknorm = masknormT + masknormNT + masknormB
    masknorm = np.transpose(masknorm, axes=(1,2,0))
    return masknorm


picts_to_add = ['TNE0006',
               'TNE0071',
               'TNE0232',
               'TNE0519',
               'TNE0798',
               'TNE0866',
               'TNE1001',
               'TNE1002',
               'TNE1080',
               'TNE1426',
               'TNE1458',
               'TNE1420',
               'TNE0886',
               'TNE0890',
               'TNE1082',
               'TNE1415',
               'TNE1418',
               'TNE1420',
               'TNE1425',
               'TNE1433',
               'TNE2097',
               'TNE2117',
               'TNE2155']
Outputdir = 'Data/NewTrainMask'
try: 
    os.mkdir(Outputdir)
except:
    print('Done')
    
MainFolder = 'Data/TestImgPredictedTiles'
for sample in os.listdir(MainFolder):
    if sample in picts_to_add:
        try: 
            os.mkdir(os.path.join(Outputdir, sample))
        except:
            print('Done')
        segmentation_l = os.listdir(os.path.join('Data/TestImgPredictedTiles', sample, 'segmentation'))
        for mask in segmentation_l:
            im_test = cv2.imread(os.path.join('Data/TestImgPredictedTiles', sample, 'segmentation', mask))
            im_test = cv2.cvtColor(im_test, cv2.COLOR_BGR2RGB)
            masknorm =  TumorNoTumorBackgroundIndex(im_test)
            masknorm[np.all(masknorm == (192, 192, 192), axis=-1)] = (150,200,150)
            masknorm[np.all(masknorm == (0, 255, 255), axis=-1)] = (195,5,2)
            masknorm[np.all(masknorm == (255, 255, 255), axis=-1)] = (255,253,252)
            masknorm = masknorm.astype('uint8')
            nname = str(mask.split('-')[0] + '.png')
            cv2.imwrite(os.path.join(Outputdir, sample, nname), cv2.cvtColor(masknorm.astype('uint8'), cv2.COLOR_RGB2BGR))