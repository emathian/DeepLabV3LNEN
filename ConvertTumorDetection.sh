## From /model/ressearch/deeplab/dataset/TumorDetection
CURRENT_DIR=$(pwd)
WORK_DIR="./TumorDetection"
PQR_ROOT="${WORK_DIR}/dataset"
SEG_FOLDER="${PQR_ROOT}/SegmentationClass"
SEMANTIC_SEG_FOLDER="${PQR_ROOT}/SegmentationClassRaw"
# Build TFRecords of the dataset.
OUTPUT_DIR="${WORK_DIR}/tfrecord"
mkdir -p "${OUTPUT_DIR}"
IMAGE_FOLDER="${PQR_ROOT}/JPEGImages"
LIST_FOLDER="${PQR_ROOT}/ImageSets"
echo "Converting Tumor detection  dataset..."
python ./build_new_tunor_detection_data.py \
  --image_folder="${IMAGE_FOLDER}" \
  --semantic_segmentation_folder="${SEMANTIC_SEG_FOLDER}" \
  --list_folder="${LIST_FOLDER}" \
  --image_format="jpg" \
  --output_dir="${OUTPUT_DIR}"