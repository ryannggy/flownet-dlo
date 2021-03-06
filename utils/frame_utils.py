import numpy as np
from os.path import *
from scipy.misc import imread
from . import flow_utils 
from numpy import newaxis

def read_gen(file_name):
    ext = splitext(file_name)[-1]
    if ext == '.png' or ext == '.jpeg' or ext == '.ppm' or ext == '.jpg':
        im = imread(file_name)
#         if im.shape[2] > 1:     #convert to using grayscale instead
#             return im[:,:,:1]
#         else:
        return im[:,:,newaxis]
    elif ext == '.bin' or ext == '.raw':
        return np.load(file_name)
    elif ext == '.flo':
        return flow_utils.readFlow(file_name).astype(np.float32)
    return []
