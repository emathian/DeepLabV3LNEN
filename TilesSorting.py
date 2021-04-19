import os
import cv2
import numpy as np
import shutil

try:
    os.mkdir('Data/NormalTissusTrainset2Extraction')
except:
    print('Done')
main_dir_correct_mask = 'Data/FullMaskTrainset2'
main_dir_correct_mask_l = os.listdir(main_dir_correct_mask)
for mask in main_dir_correct_mask_l:
    Mask_fSeg = cv2.imread(os.path.join(main_dir_correct_mask, mask))
    Mask_fSeg = cv2.cvtColor(Mask_fSeg, cv2.COLOR_BGR2RGB)
    samplename = mask.split('-')[0]
    # Cross List of tiles 
    list_tiles = os.path.join('/data/gcs/lungNENomics/work/MathianE/Tiles_512_512_fullHENorm_from1802Dir' , samplename)
    for folder in os.listdir(list_tiles):
        l_tiles = os.listdir(os.path.join(list_tiles, folder))
        for t in l_tiles:
            sample =  t.split('_')[0]
            x = int(t.split('_')[1])
            y = int(t.split('_')[2].split('.')[0])
            xf  = int(x * .25)
            yf = int(y * .25)

            xm = x + 925
            ym = y + 925

            xmf = int(xm * 0.25)
            ymf = int(ym *0.25)

            sub_mask = Mask_fSeg[xf:xmf, yf:ymf, :]
            print(sub_mask.shape)


            TumorMat = np.ones(sub_mask.shape, dtype=np.uint8) 
            TumorMat[:,:,0] = 195
            TumorMat[:,:,1] = 5
            TumorMat[:,:,2] = 2


            NoTumorMat = np.ones(sub_mask.shape, dtype=np.uint8) 
            NoTumorMat[:,:,0] = 150
            NoTumorMat[:,:,1] = 200
            NoTumorMat[:,:,2] = 150


            BackgroundMat = np.ones(sub_mask.shape, dtype=np.uint8) 
            BackgroundMat[:,:,0] = 255
            BackgroundMat[:,:,1] = 255
            BackgroundMat[:,:,2] = 255

            masknorm = np.zeros((sub_mask.shape))
            dT = np.abs(np.sum(sub_mask.astype('int8')  - TumorMat.astype('int8') , axis = 2))
            dNT = np.abs(np.sum(sub_mask.astype('int8')  - NoTumorMat.astype('int8') , axis = 2))
            dB = np.abs(np.sum(sub_mask.astype('int8')  - BackgroundMat.astype('int8') , axis = 2))
            diff_mat = np.zeros((sub_mask.shape))
            diff_mat[:,:,0] = dT
            diff_mat[:,:,1] = dNT
            diff_mat[:,:,2] = dB
            arr2dINDEX = np.argmin(diff_mat, axis = 2)

            masknormOri = np.ones((sub_mask.shape))
            masknormT = np.multiply(np.multiply(np.transpose(masknormOri, axes=(2,0,1)) ,
                                     (np.array(arr2dINDEX == 0)*1)),np.transpose(TumorMat,axes=(2,0,1)))
            masknormNT =  np.multiply(np.multiply( np.transpose(masknormOri, axes=(2,0,1)), 
                                     (np.array(arr2dINDEX == 1)*1)), np.transpose(NoTumorMat,axes=(2,0,1)))

            masknormB =  np.multiply(np.multiply(np.transpose(masknormOri, axes=(2,0,1)), 
                                     (np.array(arr2dINDEX == 2)*1)), np.transpose(BackgroundMat, axes=(2,0,1)))
            masknorm = masknormT + masknormNT + masknormB
            masknorm = np.transpose(masknorm, axes=(1,2,0))
            pTumor = np.sum(masknorm[:,:, 0] == 195) / (sub_mask.shape[0] * sub_mask.shape[1] )
            pBackground = np.sum(masknorm[:,:, 0] == 255) / (sub_mask.shape[0] * sub_mask.shape[1] )
            pNoTumor = np.sum(masknorm[:,:, 0] == 150) / (sub_mask.shape[0] * sub_mask.shape[1] )

            if pBackground < 0.5 and pNoTumor> pTumor:
                try:
                    os.mkdir(os.path.join('Data/NormalTissusTrainset2Extraction',samplename))
                except:
                    print('Done')
                shutil.copy(os.path.join(list_tiles, folder, t), os.path.join('Data/NormalTissusTrainset2Extraction',samplename, t))