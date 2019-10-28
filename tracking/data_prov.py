import sys
import numpy as np
from PIL import Image
import random
import torch
import torch.utils.data as data
import math
from utils import *


class RegionExtractor():
    def __init__(self, image, samples, crop_size, padding, batch_size, shuffle=False,train=False):

        self.image = np.asarray(image)
        self.samples = samples
        self.crop_size = crop_size
        self.padding = padding
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.train = train
        self.index = np.arange(len(samples))
        self.pointer = 0

        self.mean = self.image.mean(0).mean(0).astype('float32')

    def __iter__(self):
        return self

    def __next__(self):
        if self.pointer == len(self.samples):
            self.pointer = 0
            raise StopIteration
        else:
            next_pointer = min(self.pointer + self.batch_size, len(self.samples))
            index = self.index[self.pointer:next_pointer]
            self.pointer = next_pointer

            regions = self.extract_regions(index)
            regions = torch.from_numpy(regions)
            return regions
    next = __next__

    def extract_regions(self, index):
        regions = np.zeros((len(index),self.crop_size,self.crop_size,3),dtype='uint8')
        for i, sample in enumerate(self.samples[index]):
            regions[i] = crop_image(self.image, sample, self.crop_size, self.padding)
            if self.train:
                regions[i] = self.randomErase(regions[i])
        regions = regions.transpose(0,3,1,2).astype('float32')
        regions = regions - 128.
        return regions
    def randomErase(self,img,probability=0.5,sl = 0.02,sh=0.1,r1=0.3):
        if random.uniform(0,1)>probability:
            return img
        for attempt in range(100):
            area = img.shape[0]*img.shape[1]
            target_area = random.uniform(sl,sh)*area
            aspect_ratio = random.uniform(r1,1/r1)
            h = int(round(math.sqrt(target_area*aspect_ratio)))
            w = int(round(math.sqrt(target_area/aspect_ratio)))
            
            if w < img.shape[1] and h < img.shape[0]:
                x1 = random.randint(0,img.shape[0]-h)
                y1 = random.randint(0,img.shape[1]-w)
                img[x1:x1+h,y1:y1+w,:] = 128
                return img
        return img
        