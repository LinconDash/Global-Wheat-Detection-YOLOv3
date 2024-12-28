import os
import sys
import numpy as np
import tensorflow as tf
import albumentations as albu

from src.logger import logging
from src.exception import CustomException
from tensorflow.keras.utils import Sequence

class DataGenerator(Sequence):
    """
        Creates a data generator which is useful while providing batches of data for model training
    """
    def __init__(
                self,
                image_ids, 
                image_pixels, 
                image_labels=None, 
                shuffle=False, 
                batch_size=1, 
                transform=False):

        super().__init__()
        self.image_ids = self.image_ids
        self.image_pixels = image_pixels
        self.labels = self.image_labels
        self.shuffle = shuffle
        self.batch_size = batch_size
        self.transform = transform
        self.on_epoch_end()
        self.image_grid = self.create_image_grid()
        self.train_augmentation = None
        self.val_augmentation = None
    
    def __len__(self):
        """
            Returns the length of the dataset per batch
        """
        return int(np.floor(len(self.image_ids) / self.batch_size))

    def __getitem__(self, index):
        batch_start = index * self.batch_size 
        batch_end = (index + 1) * self.batch_size
        indeces = self.indexes[batch_start : batch_end] 
        image_ids = [self.image_ids[i] for i in indeces]
        X, Y = self.__data_generation(image_ids)
        return X, Y

    def __data_generation(self, image_ids):
        X, Y = [], []
        for image_id in image_ids:
            pixels = self.image_pixels[image_id]
            bboxes = self.labels[image_id]

            if self.transform == True:
                pixels, bboxes = self.transform_images(pixels, bboxes)
            else:
                pixels = self.contrast_image(pixels)
                bboxes = self.create_label_grid(bboxes)
            
            X.append(pixels)
            Y.append(bboxes)
        
        return np.array(X), np.array(Y)

    
