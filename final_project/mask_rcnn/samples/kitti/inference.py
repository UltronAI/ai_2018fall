import os
import sys
import time
import numpy as np
import argparse
import skimage.io
from dataset import DataLoader

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
parser.add_argument('-g', '--gpu-id', type=int, default=-1)

args = parser.parse_args()
if args.gpu_id >= 0:
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu_id)

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
    GPU_COUNT = 1
    NUM_CLASSES = 81

class InferenceConfig(KITTIConfig):
    # Set batch size to 1 since we'll be running inference on
    # one image at a time. Batch size = GPU_COUNT * IMAGES_PER_GPU
    GPU_COUNT = 1
    # BACKBONE = "resnet50"
    IMAGES_PER_GPU = 1
    DETECTION_MIN_CONFIDENCE = 0

def store_mask(mask, save_path):
    height, width, object_num = mask.shape
    img_mask = np.zeros((height, width))
    for i in range(object_num):
        mask_i = mask[:, :, i].copy().astype(int) * (i + 1)
        img_mask += mask_i
        img_mask = np.clip(img_mask, 0, i + 1)
        assert img_mask.max() <= object_num, "error occurred! {} should be less than {}".format(img_mask.max(), object_num)
    import scipy.misc
    #print(img_mask.max())
    scipy.misc.imsave(save_path, img_mask)

def makedir(path):
    if not os.path.exists(path):
        os.makedirs(path)
    return path

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

    dataloader = DataLoader(args.dataset_dir)
    # img = skimage.io.imread('/home/gaofeng/datasets/kitti/raw_data/2011_09_29/2011_09_29_drive_0004_sync/image_02/data/0000000141.png', plugin='matplotlib') * 255
    # print(img.shape)
    # masks = model.detect([img], verbose=0)[0]['masks']
    # store_mask(masks, './test.png')
    # print(masks.shape[-1])
    # exit(0)
    for date, seqname, folder, img_path in dataloader.read_raw_data():
        img = skimage.io.imread(img_path, plugin='matplotlib') * 255
        # print(img.shape)
        #if seqname.split('_')[-2] != '0026':
        #    continue
        if img.shape[-1] == 4:
            img = img[..., :3]
        #print("processing {}".format(img_path.split('/')[-1]))
        try:
            masks = model.detect([img], verbose=0)[0]['masks']
        except:
            print(img_path)
            print(img_path, file=open('output.log','a'))
        #continue
        #print(masks.shape[-1]) 
        path_ = makedir(os.path.join(args.output_dir, date))
        path_ = makedir(os.path.join(path_, seqname))
        path_ = makedir(os.path.join(path_, folder))
        path_ = makedir(os.path.join(path_, 'data'))
        save_path = os.path.join(path_, img_path.split('/')[-1])
        #print("storing")
        store_mask(masks, save_path)
        #print("done.")        

if __name__ == '__main__':
    main()
