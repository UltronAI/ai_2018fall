import sys
import os
import numpy as np
import tensorflow as tf
import skimage.color
import skimage.io
import skimage.transform
import glob as gb

class DataLoader(object):
    def __init__(self, dataset_dir, set_type='raw'):
        self.dataset_dir = dataset_dir
        self.type = set_type
    
    def read_raw_data(self):
        pass
    
