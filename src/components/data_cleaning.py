import sys
import warnings

from tqdm.auto import tqdm
from dataclasses import dataclass
from src.logger import logging
from src.exception import CustomException
from src.components.data_ingestion import DataIngestion
warnings.filterwarnings("ignore")

@dataclass
class DataCleanerConfig:
    T_threshold : int = 5
    H_threshold : int = 20000


class DataCleaner:
    def __init__(self):
        self.config = DataCleanerConfig()
        self.data = DataIngestion()
        self.train_pixels = self.data.train_pixels
        self.train_labels = self.data.train_labels
        self.val_pixels = self.data.val_pixels
        self.val_labels = self.data.val_labels

        logging.info("Cleaning training labels.")
        print("\nCleaning train data labels : ")
        self.train_labels = self.clean_labels(
            labels=self.data.train_labels,
            image_ids=self.data.train_pixels.keys(),
            tiny_area=self.config.T_threshold,
            huge_area=self.config.H_threshold
        )

        print("\n")
        logging.info("Cleaning validation labels.")
        print("Cleaning validation data labels : ")
        self.val_labels = self.clean_labels(
            labels=self.data.val_labels,
            image_ids=self.data.val_pixels.keys(),
            tiny_area=self.config.T_threshold,
            huge_area=self.config.H_threshold
        )

    def clean_labels(self, labels, image_ids, tiny_area, huge_area):
        try:
            good_labels = {}

            for image_id in tqdm(image_ids):
                good_labels[image_id] = []
                for label in labels[image_id]:
                    if label[2] * label[3] <= tiny_area:
                        continue
                    elif label[2] * label[3] > huge_area:
                        continue
                    else:
                        good_labels[image_id].append(label)
            
            return good_labels
        except Exception as e:
            raise CustomException(e, sys)