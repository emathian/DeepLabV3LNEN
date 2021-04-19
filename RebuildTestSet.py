import os
import cv2
import numpy as np
getTrain2list = os.listdir('models/research/deeplab/datasets/TumorDetection/dataset/JPEGImages')
sample = []
path2oriimg = '/data/gcs/lungNENomics/work/MathianE/FullSlidesToJpegHENormHighQuality'
for ele in getTrain2list:
    csample = ele.split('_')[0]
    if csample not in sample and csample.find('TNE') != -1:
        sample.append(csample)

try:
    os.mkdir('Data/FullMaskTrainset2')
except:
    print('Done')
outputdir  = 'Data/FullMaskTrainset2'
getTrain2list = os.listdir('models/research/deeplab/datasets/TumorDetection/dataset/SegmentationClass')
path2oriimg = '/data/gcs/lungNENomics/work/MathianE/FullSlidesToJpegHENormHighQuality'
for ele in sample:
    print(ele)
    csample = ele
    ori_im = cv2.imread(os.path.join(path2oriimg, csample + '.jpg'))
    OverlayTot = np.zeros(ori_im.shape)
    tiles_list = []
    for tile in getTrain2list:
        if tile.split('_')[0]  == csample:
            print('tile ', tile)
            tiles_list.append(tile)
    for t in tiles_list:
        sub_pict = cv2.imread(os.path.join('models/research/deeplab/datasets/TumorDetection/dataset/SegmentationClass',t))
        sample_x =  int(t.split('_')[1])
        sample_y =  int(t.split('_')[2].split('.')[0])
        OverlayTot[sample_x:sample_x+512, 
                  sample_y:sample_y+512,] = sub_pict
    outputfname = '{}-segmap.jpg'.format(csample)
    print('outputfname ',outputfname)
    cv2.imwrite(os.path.join(outputdir, outputfname),OverlayTot)
