# DeepLabV3LNEN
Semantic segmentation algorithm with DeepLabV3+.
**Environment : Tensorflow 1.12**
## Build the mask
+ 1 - Convert the WSI to 1/8 jpeg images:
    - `\ImgProcessing\FullSlidesToJpeg\FullSlidesToJpegNf.py` 
        - Args: 
             - inputdir : path to original WSI directory;
             - inputfile : slide file name
             - outputdir : directory where the jpeg will be saved
    - `\ImgProcessing\FullSlidesToJpeg\NfFullSlidesToJpeg.nf` 
    - `\ImgProcessing\FullSlidesToJpeg\SlurmNfFullSlidesToJpecg.sh`
    - Current images: `\data\gcs\lungNENomics\work\MathianE\FullSlidesToJpegHighQuality` 332 images
+ 2 - Vahadane's Color normalisation of full slides images 
    - `mathiane\ImgProcessing\Vahadane\HE_NormFullSlides.py`
        - Args: 
             - inputdir : Directory where the jpeg images are stored;
             - inputfile : slide file name
             - outputdir : directory where the jpeg will be saved
    - `mathiane\ImgProcessing\Vahadane\NfFullHE.nf`
    - Ref img: '/TNE0287.jpg'
    
+ 3 - Labeling with QuPath:
    - 1: Local drectory : 'C:\Users\mathiane\OneDrive - IARC\Desktop\TumorDetectionFromHEJpeg'
    - 2: Train a pixels classifier
    - 3: export the model
    - 4: predict
    - 5: export the prediction: 'MasktumorNoTumor.groovy'

    
+ 4 - Tiling WSI 512x512:
     - Args: 
         - inputdir : path to original WSI directory;
         - outputdir : where the tiles are stored: outputdir/sample_id/x.jpeg
    - Copy the tiles to: 'SemanticSegmentation/models/research/deeplab/datasets/TumorDetection/dataset/SegmentationClass'
+ 5 - Image indexing: transform 3 channel mask to index s.t.:
    - palette = {(195,   5,   2) : 0 , # Tumor
         (150,  200, 150) : 1, # No Tumor
         (252,253,252) : 2 # Background }
    - Outputdir : '/home/mathiane/SemanticSegmentation/models/research/deeplab/datasets/TumorDetection/dataset/SegmentationClassRaw'

## Train the network
### Folder organisation:

+ models\research\deeplab
    + datasets\
        + Run: ConvertTumorDetection.sh create tf.record
        + \TumorDetection
            + \dataset
                + ImageSets:
                    + train.txt // LIst of tiles names without extension
                    + val.txt
                    + travail.txt 
                + JPEFImages:
                    + Tiles // Folder with all samples' tiles 
                    - Nb: jpg format
                + SegmentationClass:
                    + All samples' masks 
                    - Nb: png
                + SegmentationClassRae:
                    + All samples' masls indexed
                    - Nb: png
            + \tfrecord
            + exp
                + train_on_trainvalset
                    + eval: Checkpoints
                    + frozen_graph
                    + inits_models : Path to the initial graph
                    + train: checkpoints
                    + vis: Get inference on one slide
                    
### Run!
+ `SemanticSegmentation\models\research\train-tumor-detection.sh`
+ **Warning!** Update the paths 
### Eval 
+ `SemanticSegmentation\models\research\eval-tumor-detection.sh`
+ **Warning!** Update the paths 
### Visualize:
+ `SemanticSegmentation\models\research\vis-tumor-detection.sh`
+ **Warning!** Update the paths 
### Export the graph
+ `SemanticSegmentation\models\research\export-tumor-detection.sh`
+ **Warning!** Update the paths
## Predict
+  `SemanticSegmentation\models\research\test_inference.py`
- Args: 
     - inputdir : path to the tiles normalised to predict;
     - outputdir : directory where the infered masks will be stored
     - path2frozengraph: Path to the frozen graph .tar
- Generate the mask s.t.:
    `FULL_COLOR_MAP = np.array([[0,255,255],#Tumor
                          [192,192, 192], #NoTumor
                          [255, 255, 255]])# Backgroud
                          #label_to_color_image(FULL_LABEL_MAP)`

- Generate the overlay
- Organisation of the prediction:
    - outputdir\
        - sample_id\
            -segmentation\
                - tilename-segmap.jpg
            - overlay\ 
                - tilename-overlay.jpg
## PostProcess
+ Output overview:
    +  `\SemanticSegmentation\REbuildTestSet.py`
    + **Warning!** Update the paths 
    + Either the overlay or the segmap
+ Tiles sortng: Sort the original tiles (not from full jpeg normalised WSI) according to the prediction.
    + 'TilesSorting.ipynb' // Nb pixels Background > .5% or Nb pixels NoTumor > Nb pixels NoTumor == reject
    + Tricks to match the infered tile with the original one.    

## New training loop
- Convert infered masks to trained mask
- `InfMaskToTrainMask.py`
- Args:
    - List of pictures to add to the next training process
    - Outputdir: Convert  :
        - [[0,255,255],#Tumor
        - [192,192, 192], #NoTumor
        - [255, 255, 255]])# Backgroud
        to: 
        - palette = {(195,   5,   2) : 0 , # Tumor
         (150,  200, 150) : 1, # No Tumor
         (252,253,252) : 2 # Background }
## Note:
- Careful to images format from jpg to png channel values differ
- Careful to images' and masks' name


                          
