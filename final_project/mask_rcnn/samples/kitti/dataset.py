import sys
import os, glob
import numpy as np
import tensorflow as tf
import skimage.color
import skimage.io
import skimage.transform

class DataLoader(object):
    def __init__(self, dataset_dir):
        self.dataset_dir = dataset_dir
        self.imglist = []
        self.read_raw_data()

    def read_raw_data(self):
        for d in glob.glob(self.dataset_dir, '/*/')
            data = d.split('/')[-2]
            for d2 in glob.glob(d + '*/'):
                seqname = d2.split('/')[-2]
                for subfolder in ['image_02/data', 'image_03/data']:
                    folder = d2 + subfolder
                    files = glob.glob(folder + '/*.png')
                    files = [file for file in files if not 'disp' in file and not 'flip' in file and not 'seg' in file]
                    files = sorted(files)
                    self.imglist += files

