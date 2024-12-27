import os
import sys
import pandas as pd
import numpy as np
import warnings

from dataclasses import dataclass
from tqdm.auto import tqdm
from src.logger import logging
from src.exception import CustomException
from src.utils import str_to_float, load_image_as_pixels
warnings.filterwarnings("ignore")

@dataclass
class DataIngestionConfig:
    train_dir : str = os.path.join("artifacts", "train")
    test_dir : str = os.path.join("artifacts", "test")
    train_labels : str = os.path.join("artifacts", "train.csv")

class DataIngestion:
    def __init__(self):
        self.config = DataIngestionConfig()
        self.labels = self.initiate_dataingestion()
        self.train_pixels, self.train_labels, self.val_pixels, self.val_labels = self.train_validation_split(
            dirname=self.config.train_dir, 
            labels=self.labels,
            split_size=0.8)
    
    def initiate_dataingestion(self):
        try:
            logging.info("Initiating Data Ingestion.")
            labels = pd.read_csv(self.config.train_labels)
            new_labels = labels.groupby("image_id").apply(str_to_float)
            logging.info("Converted the bboxes from str to float values.")
            return new_labels
        except Exception as e:
            raise CustomException(e, sys)

    def train_validation_split(self, dirname, labels, split_size):
        try:
            logging.info(f"Applying train-val split with split size of {split_size}.")
            all_image_ids = list(labels.index)
            train_size = int(np.floor(split_size * len(all_image_ids)))
            train_image_ids = all_image_ids[:train_size]
            val_image_ids = all_image_ids[train_size:]

            train_pixels = {}
            val_pixels = {}
            train_labels = {}
            val_labels = {}

            print("Loading train image pixels and labels : ")
            for image_id in tqdm(train_image_ids):
                train_pixels[image_id] = load_image_as_pixels(file_path=dirname, image_id=image_id)
                train_labels[image_id] = labels[image_id].copy() / 4  # because we will work with 256 X 256 image data 
            
            print("\n")

            print("Loading validation image pixels and labels : ")
            for image_id in tqdm(val_image_ids):
                val_pixels[image_id] = load_image_as_pixels(file_path=dirname, image_id=image_id)
                val_labels[image_id] = labels[image_id].copy() / 4

            logging.info(f"Converted train-images to train and validataion images and labels.")
            return train_pixels, train_labels, val_pixels, val_labels
        except Exception as e:
            raise CustomException(e, sys)
