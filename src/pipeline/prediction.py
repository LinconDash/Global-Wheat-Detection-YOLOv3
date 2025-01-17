import os 
import sys
import numpy as np
import albumentations as albu
import tensorflow as tf
import warnings

from PIL import Image, ImageDraw
from tqdm.auto import tqdm
from src.components.model import YOLOv3
from src.components.loss import YOLOLoss
from src.components.data_generator import DataGenerator
from src.exception import CustomException
from src.logger import logging
from src.utils import process_predictions, draw_bbox

warnings.filterwarnings("ignore")

# HYPERPARAMETERS
MODEL_NAME = "yolo_model.h5"
SAVE_DIR = "./models"
VAL_BATCH_SIZE = 5
PRED_DIR = "./predictions"
TEST_IMAGE_DIR = "./artifacts/test"

def predict():
    model_path = os.path.join(SAVE_DIR, MODEL_NAME)
    if os.path.exists(model_path):
        print("Trained model found.")
        print("\n")
        print("Loading the trained model.")
        model = tf.keras.models.load_model(model_path, custom_objects={"YOLOv3":YOLOv3})
        print("Trained Model loaded successfully.")
    else:
        raise FileNotFoundError(f"No such model exists at : {model_path}")

    # Load and Predict test data
    logging.info("Predicting test data")
    test_predictions = []
    test_image_ids = os.listdir(TEST_IMAGE_DIR)
    test_image_ids = [image_id[:-4] for image_id in test_image_ids]

    print("Predicting bounding boxes of the test images.")
    for i, image_id in tqdm(enumerate(test_image_ids)):
        image = Image.open(TEST_IMAGE_DIR + "/" + image_id + ".jpg")
        image = image.resize((256, 256))
        pixels = np.asarray(image)
        
        # same as val_augmentations
        augmentations = albu.Compose([
            albu.CLAHE(p=1),
            albu.ToGray(p=1)
        ])

        aug_result = augmentations(image=pixels)
        pixels = np.array(aug_result["image"]) / 255
        pixels = np.expand_dims(pixels, axis=0)

        bboxes = model.predict(pixels)
        test_predictions.append(bboxes)
    

    data_gen = DataGenerator()
    image_grid = data_gen.image_grid
    test_predictions = np.concatenate(test_predictions)
    test_predictions = process_predictions(test_predictions, test_image_ids, image_grid)
    
    os.makedirs(PRED_DIR, exist_ok=True)
    logging.info(f"Saving the predictions in {PRED_DIR} directory")
    print(f"Saving the predictions in {PRED_DIR} directory.")
    for i, image_id in tqdm(enumerate(test_image_ids)):
        image = Image.open(TEST_IMAGE_DIR + "/" + image_id + ".jpg")
        image = image.resize((256, 256))
        draw = ImageDraw.Draw(image)
        bboxes = test_predictions[image_id]
        for bbox in bboxes:
            draw_bbox(draw, bbox)
        image.save(PRED_DIR + "/" + image_id + ".jpg")
    print(f"All the predictions are saved successfully.")


if __name__ == "__main__":
    predict()