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
TumorMat[:,:,0] = 195
TumorMat[:,:,1] = 5
TumorMat[:,:,2] = 2


NoTumorMat = np.ones((512,512,3), dtype=np.uint8) 
NoTumorMat[:,:,0] = 150
NoTumorMat[:,:,1] = 200
NoTumorMat[:,:,2] = 150


BackgroundMat = np.ones((512,512,3), dtype=np.uint8) 
BackgroundMat[:,:,0] = 252
BackgroundMat[:,:,1] = 253
BackgroundMat[:,:,2] = 252


palette = {(195,   5,   2) : 0 , # Tumor
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


def convert_from_color_segmentation(arr_3d):
    arr_2d = np.zeros((arr_3d.shape[0], arr_3d.shape[1]), dtype=np.uint8)

    for c, i in palette.items():
        m = np.all(arr_3d == np.array(c).reshape(1, 1, 3), axis=2)
        arr_2d[m] = i
    return arr_2d

label_dir ='/home/mathiane/SemanticSegmentation/models/research/deeplab/datasets/TumorDetection/dataset/SegmentationClass'
new_label_dir = '/home/mathiane/SemanticSegmentation/models/research/deeplab/datasets/TumorDetection/dataset/SegmentationClassRaw'

if not os.path.isdir(new_label_dir):
	print("creating folder: ",new_label_dir)
	os.mkdir(new_label_dir)
else:
	print("Folder alread exists. Delete the folder and re-run the code!!!")

label_files = os.listdir(label_dir)

for l_f in tqdm(label_files):
    
    im = cv2.imread(os.path.join(label_dir, l_f))
    im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
    masknorm =  TumorNoTumorBackgroundIndex(im)
    arr_2d = convert_from_color_segmentation(masknorm)
    Image.fromarray(arr_2d).save(os.path.join(new_label_dir , l_f))
   