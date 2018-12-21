import os
import sys
import time
import numpy as np
import argparse
import skimage.io

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Root directory of the project
ROOT_DIR = os.path.abspath("../../")

# Import Mask RCNN
sys.path.append(ROOT_DIR)  # To find local version of the library
from mrcnn.config import Config
from mrcnn import model as modellib, utils

# Parse command line arguments
parser = argparse.ArgumentParser(description="Do instance segmentation using Mask RCNN on KITTI datasets",
            formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--dataset-dir', type=str, required=True, default=None, metavar='/path/to/kitti/datasets')
parser.add_argument('--output-dir', type=str, required=True, default='output', metavar='/path/to/store/output/masks')
parser.add_argument('--model', type=str, required=True, default='./mask_rcnn_balloon.h5', metavar='/path/to/pretrained/model')
parser.add_argument('--log-dir', type=str, default='log', metavar='/path/to/log')
args = parser.parse_args()

class KITTIConfig(Config):
    """Configuration for training on MS COCO.
    Derives from the base Config class and overrides values specific
    to the KITTI dataset.
    """
    # Give the configuration a recognizable name
    NAME = "kitti"

    # We use a GPU with 12GB memory, which can fit two images.
    # Adjust down if you use a smaller GPU.
    IMAGES_PER_GPU = 2

    # Uncomment to train on 8 GPUs (default is 1)
    # GPU_COUNT = 8

    # Number of classes (including background)
    NUM_CLASSES = 1 + 80  # COCO has 80 classes

class InferenceConfig(KITTIConfig):
    # Set batch size to 1 since we'll be running inference on
    # one image at a time. Batch size = GPU_COUNT * IMAGES_PER_GPU
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1
    DETECTION_MIN_CONFIDENCE = 0

def main():
    # prepare configure
    print("=> preparing configure for mask rcnn")
    config = InferenceConfig()
    config.display()

    if not os.path.exists(args.log_dir):
        os.makedirs(args.log_dir)
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    # create model
    print("=> creating mask rcnn model")
    model = modellib.MaskRCNN(mode='inference', config=config, model_dir=args.log_dir)
    print("=> loading weights from {}".format(args.model))
    model.load_weights(args.model, by_name=True)

    # TODO: implement the dataloader
    dataloader = None
    # for i, img in enumerate(dataloader):
    img_path = '1.png'
    img = skimage.io.imread(img_path)
    if img.shape[-1] == 4:
        img = img[..., :3]
    masks = model.detect([img], verbose=0)[0]['masks']
    print(np.array(masks).shape)
    #print(type(masks))

if __name__ == '__main__':
    main()
