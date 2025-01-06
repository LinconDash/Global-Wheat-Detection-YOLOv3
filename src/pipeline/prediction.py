import os 
import sys
import tensorflow as tf

from src.components.model import YOLOv3
from src.components.loss import YOLOLoss
from src.components.data_cleaning import DataCleaner
from src.components.data_generator import DataGenerator
from src.exception import CustomException
from src.logger import logging
from src.utils import (
    process_predictions,
    save_predictions
)

# HYPERPARAMETERS
MODEL_NAME = "yolo_model.h5"
SAVE_DIR = "./models"
VAL_BATCH_SIZE = 5
PRED_DIR = "./predictions"

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

    # Load data
    data = DataCleaner()
    val_pixels = data.val_pixels
    val_labels = data.val_labels
    val_image_ids = list(val_pixels.keys())

    val_generator = DataGenerator(
        image_ids=val_image_ids,
        image_pixels=val_pixels,
        image_labels=val_labels,
        shuffle=False,
        batch_size=VAL_BATCH_SIZE,
        transform=False
    )

    image_grid = val_generator.image_grid 

    # Predictions
    print("Predicting the bounding boxes.")
    val_predictions = model.predict(val_generator)
    val_predictions = process_predictions(val_predictions, val_image_ids, image_grid)
    os.makedirs(PRED_DIR, exist_ok=True)
    print(f"Saving predictions in {PRED_DIR} directory.")
    save_predictions(val_predictions, val_image_ids, val_pixels, PRED_DIR)
    print("Predictions saved successfully.") 

if __name__ == "__main__":
    predict()