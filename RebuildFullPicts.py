import os 
import cv2
import numpy as np
import argparse
parser = argparse.ArgumentParser(description='Test set carcinoids for scanners.')
parser.add_argument('--inputdir', type=str,    help="Input directory where the images are stored")
parser.add_argument('--outputdir', type=str,    help='output directory where the files will be stored')
parser.add_argument('--path2oriimg', type=str,    help='output directory where the files will be stored')

args = parser.parse_args()
outputdir = args.outputdir
inputdir = args.inputdir
path2oriimg =  args.path2oriimg
try:
    os.mkdir(outputdir)
except:
    print('Outputdir already created')
for sample in os.listdir(inputdir):
    ImgsList = os.listdir(os.path.join(inputdir, sample, 'segmentation'))
    ori_im = cv2.imread(os.path.join(path2oriimg, sample + '.jpg'))
    OverlayTot = np.zeros(ori_im.shape)
    for img in ImgsList:
        sub_pict = cv2.imread(os.path.join(inputdir, sample, 'segmentation', img))
        sample_x =  int(img.split('_')[1])
        sample_y =  int(img.split('_')[2].split('-')[0])
        OverlayTot[sample_x:sample_x+512, 
                  sample_y:sample_y+512,] = sub_pict
    outputfname = '{}-segmap.jpg'.format(sample)
    cv2.imwrite(os.path.join(outputdir, outputfname),OverlayTot)
