from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys
import cv2
import numpy as np
import argparse
from dataset import DataLoader

import torch
import torch.nn as nn

from modeling.model_builder import Generalized_RCNN
from core.config import cfg, cfg_from_file, assert_and_infer_cfg
import utils.net as net_utils
from core.test import im_detect_all

# Parse command line arguments
parser = argparse.ArgumentParser(description="Do instance segmentation using Mask RCNN on KITTI datasets",
            formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--dataset-dir', type=str, required=True, default=None, metavar='/path/to/kitti/datasets')
parser.add_argument('--output-dir', type=str, required=True, default='output', metavar='/path/to/store/output/masks')
parser.add_argument('--ckpt', type=str, required=True, default=None, metavar='/path/to/pretrained/model')
parser.add_argument('--cfg', dest='cfg_file', required=True, help='optional config file')
parser.add_argument('-g', '--gpu-id', type=int, default=-1)

args = parser.parse_args()
if args.gpu_id >= 0:
    args.cuda = True
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu_id)

# OpenCL may be enabled by default in OpenCV3; disable it because it's not
# thread safe and causes unwanted GPU memory allocations.
cv2.ocl.setUseOpenCL(False)

def makedir(path):
    if not os.path.exists(path):
        os.makedirs(path)
    return path

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

def main():
    assert torch.cuda.is_available(), "Need a CUDA device to run the code."
    assert args.ckpt, "Need a pretrained model to do inference."

    print("=> called with args:")
    print(args)

    print("=> loading cfg from file {}".format(args.cfg_file))
    cfg_from_file(args.cfg_file)
    cfg.MODEL.LOAD_IMAGENET_PRETRAINED_WEIGHTS = False  # Don't need to load imagenet pretrained weights
    cfg.NUM_GPUS = 1
    assert_and_infer_cfg()

    print("=> creating Mask RCNN model")
    model = Generalized_RCNN()
    if args.cuda:
        model = model.cuda()

    print("=> loading pretrained model")
    checkpoint = torch.load(args.ckpt, map_location=lambda storage, loc: storage)
    net_utils.load_ckpt(model, checkpoint['model'])
    model.eval()

    print("=> preparing data loader")
    dataloader = DataLoader(args.dataset_dir)
    for date, seqname, folder, img_path in dataloader.read_raw_data():
        img = cv2.imread(img_path)
        masks = im_detect_all(model, img)
        exit(0)

        path_ = makedir(os.path.join(args.output_dir, date))
        path_ = makedir(os.path.join(path_, seqname))
        path_ = makedir(os.path.join(path_, folder))
        path_ = makedir(os.path.join(path_, 'data'))
        save_path = os.path.join(path_, img_path.split('/')[-1])
        store_mask(masks, save_path)


if __name__ == '__main__':
    main()