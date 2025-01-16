import os
import sys
import numpy as np
import tensorflow as tf
import warnings
import albumentations as albu

from src.logger import logging
from src.exception import CustomException
from tensorflow.keras.utils import Sequence
warnings.filterwarnings("ignore")

class DataGenerator(Sequence):
    """
        Creates a data generator which is useful while providing batches of data for model training
    """
    def __init__(
                self,
                image_ids=None, 
                image_pixels=None, 
                image_labels=None, 
                shuffle=False, 
                batch_size=1, 
                transform=False):

        super().__init__()
        logging.info("Creating Data Generator.")
        self.image_ids = image_ids
        self.image_pixels = image_pixels
        self.labels = image_labels
        self.shuffle = shuffle
        self.batch_size = batch_size
        self.transform = transform
        if self.image_ids is not None:
            self.on_epoch_end()
        self.image_grid = self.create_image_grid()

        # These are the data augmentation techniques for the training data 
        self.train_augmentation = albu.Compose([
            albu.RandomSizedCrop(
                min_max_height=(200, 200), 
                size=[256, 256], 
                p=0.8
            ),
            albu.OneOf([
                albu.HueSaturationValue(),
                albu.RandomBrightnessContrast()
            ], p=1),
            albu.OneOf([
                albu.HorizontalFlip(),
                albu.RandomRotate90(),
            ], p=1),
            albu.CLAHE(p=1),
            albu.ToGray(p=1)
        ], 
        bbox_params={'format': 'coco', 'label_fields': ['labels']})     # using format coco due to originally given

        # These are the data augmentation techniques for the validation data or if transform is False
        self.val_augmentation = albu.Compose([
            albu.CLAHE(p=1),
            albu.ToGray(p=1),
        ])
    
    def __len__(self):
        """
            Returns the length of the dataset per batch
        """
        try:
            return int(np.floor(len(self.image_ids) / self.batch_size))
        except Exception as e:
            raise CustomException(e, sys)

    def __getitem__(self, index):
        """
            Returns the indexed image and its corresponding label from a batch
        """
        try:
            batch_start = index * self.batch_size 
            batch_end = (index + 1) * self.batch_size
            indeces = self.indexes[batch_start : batch_end] 
            image_ids = [self.image_ids[i] for i in indeces]
            X, Y = self.__data_generation(image_ids)
            return X, Y
        except Exception as e:
            raise CustomException(e, sys)

    def __data_generation(self, image_ids):
        """
            Helper function for data generation from the indexed batches and 
            convert into numpy array after applying transformation.
            Before this function : labels = (N, 4)
            After this function : labels = (N, 32, 32, 10)
        """
        try:
            X, Y = [], []
            for image_id in image_ids:
                pixels = self.image_pixels[image_id]
                bboxes = self.labels[image_id]

                if self.transform == True:
                    pixels, bboxes = self.transform_images(pixels, bboxes)
                else:
                    pixels = self.contrast_images(pixels)
                    bboxes = self.create_label_grid(bboxes)
                
                X.append(pixels)
                Y.append(bboxes)
            
            return np.array(X), np.array(Y)
        except Exception as e:
            raise CustomException(e, sys)

    def on_epoch_end(self):
        try:
            self.indexes = np.arange(len(self.image_ids))
            if self.shuffle is True:
                np.random.shuffle(self.indexes)
        except Exception as e:
            raise CustomException(e, sys)

    def contrast_images(self, pixels):
        """
            Returns only simple transforms in pixels of image and no transformation in labels
        """
        try:
            aug_result = self.val_augmentation(image=pixels)
            return np.array(aug_result["image"] / 255)
        except Exception as e:
            raise CustomException(e, sys)

    def transform_images(self, pixels, bboxes):
        """
            Returns the predefined transforms in pixels of image and also in labels,
            Also creates a label grid of (32, 32, 10) w.r.t image grid (32, 32)
        """
        try:
            bbox_labels = np.ones(len(bboxes))
            aug_result = self.train_augmentation(image=pixels, bboxes=bboxes, labels=bbox_labels)
            bboxes = self.create_label_grid(aug_result["bboxes"])
            pixels = np.array(aug_result["image"] / 255)
            return pixels, bboxes
        except Exception as e:
            raise CustomException(e, sys)

    def create_image_grid(self):
        """
            Converts the image into grid of 32 X 32 for the YOLO Algorithm,
            where each cell is used to determine the coordinates with respect to each cell. 
        """
        try:
            image_grid = np.zeros((32, 32, 4))
            # Since we divide the image in 32 X 32 cells, we assign first cell as (0, 0, 256/32, 256/32)
            cell = [0, 0, 256/32, 256/32]
            for i in range(32):
                cell[0] = 0
                for j in range(32):
                    image_grid[i][j] = cell
                    cell[0] = cell[0] + cell[2]
                cell[1] = cell[1] + cell[3]
            return image_grid
        except Exception as e:
            raise CustomException(e, sys)

    def create_label_grid(self, bboxes):
        """
            Converts the label format from left-upper point and height-width
            to midpoint and height-width format by assigning bbox to each grid cell
            of the image from the actual labels stored in a dictionary.
            Each cell will contain an array of size 10 i.e of 2 anchor boxes coordinates 
            and thier probabilities.
        """
        try:
            label_grid = np.zeros((32, 32, 10))
            for i in range(32):
                for j in range(32):
                    cell = self.image_grid[i][j]
                    label_grid[i][j] = self.create_anchor_boxes(cell, bboxes)
            return label_grid
        except Exception as e:
            raise CustomException(e, sys)

    def create_anchor_boxes(self, cell, bboxes):
        """
            Create two anchor boxes if two bbox are present in the same cell
            and changes the format into a yolo format i.e mid-point and height-width format
        """
        try:
            x, y, width, height = cell
            x_max = x + width
            y_max = y + height

            anchor_1 = np.zeros((5))
            anchor_2 = np.zeros((5))

            for bbox in bboxes:
                bbox_x, bbox_y, bbox_w, bbox_h = bbox
                bbox_mid_x = bbox_x + (bbox_w / 2)
                bbox_mid_y = bbox_y + (bbox_h / 2)
                if x <= bbox_mid_x < x_max and y <= bbox_mid_y < y_max:
                    if anchor_1[0] == 0:
                        anchor_1 = self.change_box_format(cell, bbox)
                    elif anchor_2[0] == 0:
                        anchor_2 = self.change_box_format(cell, bbox)
                    else:
                        break
            return np.concatenate((anchor_1, anchor_2), axis=None)    
        except Exception as e:
            raise CustomException(e, sys)        

    def change_box_format(self, cell, bbox):
        """
            Helper function for changing the format of anchor boxes with relative to each cell
        """
        try:
            cell_x, cell_y, cell_width, cell_height = cell
            box_x, box_y, box_width, box_height = bbox

            # center the box_x and box_y first 
            box_x = box_x + (box_width / 2)
            box_y = box_y + (box_height / 2)

            # Make it relative to the cell
            box_x = (box_x - cell_x) / cell_width
            box_y = (box_y - cell_y) / cell_height

            # adjust the height and width of box 
            box_height = box_height / 256
            box_width = box_width / 256

            return [1, box_x, box_y, box_width, box_height]
        except Exception as e:
            raise CustomException(e, sys)