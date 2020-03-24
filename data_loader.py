#import scipy
import os
import time
from glob import glob
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

class DataLoader():
    def __init__(self, dataset_name, img_res=(128, 128)):
        self.dataset_name = dataset_name
        self.img_res = img_res
        np.random.seed(42)

#    def load_data(self, batch_size=10, is_testing=False):
#       
#        data_type = "train" if not is_testing else "test"
#        path_A = glob('./datasets/%s/%s/features/*' % (self.dataset_name, data_type))
#        path_B = glob('./datasets/%s/%s/targets/*' % (self.dataset_name, data_type))
#        
#        image_indexes = np.random.random_integers(0, high=len(path_A), size=batch_size)
#
#        imgs_A = []
#        imgs_B = []
#        for i in range(len(image_indexes)):
#            img_A = self.imread(path_A[i])
#            img_B = self.imread(path_B[i])
#            
#            img_A = np.array(img_A.resize(self.img_res))
#            img_B = np.array(img_B.resize(self.img_res))
#            
#            imgs_A.append(img_A)
#            imgs_B.append(img_B)
#
#        #normalize
#        imgs_A = np.array(imgs_A)/127.5 - 1.
#        imgs_B = np.array(imgs_B)/127.5 - 1.
#
#        return imgs_A, imgs_B

    def load_batch(self, batch_size=1, is_testing=False):
        data_type = "train" if not is_testing else "test"
        path_A = sorted(glob('./datasets/%s/%s/features/*' % (self.dataset_name, data_type)))
        path_B = sorted(glob('./datasets/%s/%s/targets/*' % (self.dataset_name, data_type)))
        
        #np.random.seed(int(time.time()))
        image_indexes = np.random.random_integers(0, high=len(path_A), size=batch_size)
       # print(path_A[image_indexes[1]])
        imgs_A = []
        imgs_B = []
        for i in range(len(image_indexes)):

            
            img_A = self.imread(path_A[image_indexes[i]-1])
            img_B = self.imread(path_B[image_indexes[i]-1])

            img_A = np.array(img_A.resize(self.img_res))
            img_B = np.array(img_B.resize(self.img_res))

            img_A = img_A[:,:,0:3]
            img_B = img_B[:,:,0:3]

            imgs_A.append(img_A)
            imgs_B.append(img_B)

        #normalize
        imgs_A = np.array(imgs_A)/127.5 - 1.
        imgs_B = np.array(imgs_B)/127.5 - 1.

        return imgs_A, imgs_B
    
    def getSetLen(self, batch_size=1, is_testing=False):
        data_type = "train" if not is_testing else "test"
        path_A = glob('./datasets/%s/%s/features/*' % (self.dataset_name, data_type))
        return len(path_A)
    
    
    def imread(self, path):
     #   return scipy.misc.imread(path, mode='RGB').astype(np.float)
        return Image.open(path)
