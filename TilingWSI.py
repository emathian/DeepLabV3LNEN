import os
import numpy
import cv2
import argparse
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Test set carcinoids for scanners.')
    parser.add_argument('--inputdir', type=str,    help="Input directory where the images are stored")
    parser.add_argument('--outputdir', type=str,    help='output directory where the files will be stored')
    args = parser.parse_args()
    outputdir = args.outputdir
    inputdir = args.inputdir
    try:
        os.mkdir(outputdir)
    except:
        print('Outputdir already exist ', outputdir)
    l_imgs_to_tile = os.listdir(inputdir)
    for img in l_imgs_to_tile:
        sample_id = img.split('.')[0]
        try:
            os.mkdir(os.path.join(outputdir, sample_id))
        except:
            print('Outputdir already exist ', outputdir, sample_id)
        cpath = os.path.join(outputdir,sample_id)
        im = cv2.imread(os.path.join(inputdir, img))
        lx = list(range(0,im.shape[0], 512))
        ly = list(range(0, im.shape[1], 512))
        lastx = lx[-1] - (512 - im.shape[0] % 512)
        lasty = ly[-1] - (512 - im.shape[1] % 512)
        lx.append(lastx)
        ly.append(lasty)

        for xi in range(len(lx)):
            for yi in range(len(ly)):
                #print('xi ', xi, ' yi ', yi)
                if xi != len(lx) -2 and yi != len(ly) -2:
#                     print('xi ', xi, ' yi ', yi)
                    if xi <= len(lx) -3 and yi <= len(ly) -3:
                        xmin = lx[xi]
                        xmax = lx[xi+1]
                        ymin = ly[yi]
                        ymax = ly[yi+1]
                        ctile = im[xmin:xmax, ymin:ymax, :]
                        cv2.imwrite( os.path.join( cpath,\
                                                  ("{}_{}_{}.jpg").format(sample_id, xmin, ymin )), ctile)
                    elif xi == len(lx) -1  and yi <= len(ly) - 3:
                        xmin = lx[xi]
                        xmax = im.shape[0]
                        ymin = ly[yi]
                        ymax = ly[yi+1]
                        ctile = im[xmin:xmax, ymin:ymax, :]
                        cv2.imwrite( os.path.join( cpath,\
                                                  ("{}_{}_{}.jpg").format(sample_id, xmin, ymin )), ctile)

                    elif xi <= len(lx) - 3 and yi == len(ly) -1 :
                        xmin = lx[xi]
                        xmax = lx[xi+1]
                        ymin = ly[yi]
                        ymax = im.shape[1]
                        ctile = im[xmin:xmax, ymin:ymax, :]
                        cv2.imwrite( os.path.join( cpath,\
                                                  ("{}_{}_{}.jpg").format(sample_id, xmin, ymin )), ctile)

                    elif  xi == len(lx) -1 and yi == len(ly) -1:
                        xmin = lx[xi]
                        xmax = im.shape[0]
                        ymin = ly[yi]
                        ymax = im.shape[1]
                        ctile = im[xmin:xmax, ymin:ymax, :]
                        cv2.imwrite( os.path.join( cpath,\
                                                  ("{}_{}_{}.jpg").format(sample_id, xmin, ymin )),ctile)
                else:
                    print(" Skip {} {} ".format(lx[xi], ly[yi]), xi , yi)
