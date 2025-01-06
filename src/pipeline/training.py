import os
import sys
import numpy as np
import warnings
import tensorflow as tf

from tensorflow import keras
from tensorflow.keras.optimizers import Adam

from src.components.data_cleaning import DataCleaner
from src.components.data_generator import DataGenerator
from src.components.loss import YOLOLoss
from src.components.model import YOLOv3

from src.utils import save_model
from src.exception import CustomException
from src.logger import logging

from tensorflow.keras.callbacks import (
    EarlyStopping,
    ReduceLROnPlateau,
    ModelCheckpoint
)

warnings.filterwarnings("ignore")

# HYPERPARAMETERS
TRAIN_BATCH_SIZE = 10
VAL_BATCH_SIZE = 5
EPOCHS = 10
LEARNING_RATE = 1e-4
CHECKPOINTS_DIR = "./checkpoints"
SAVE_DIR = "./models"
MODEL_NAME = "/yolo_model.h5"

def train():
    # Data Ingestion and Cleaning:
    data = DataCleaner()
    train_pixels = data.train_pixels
    train_labels = data.train_labels
    val_pixels = data.val_pixels
    val_labels = data.val_labels

    print("Data Cleaned Successfully.\n\n")
    
    # Data Generator:
    print("Creating Data generators.\n\n")
    train_image_ids = list(train_pixels.keys())
    val_image_ids = list(val_pixels.keys())

    train_generator = DataGenerator(
        train_image_ids,
        train_pixels,
        train_labels,
        shuffle=True,
        batch_size=TRAIN_BATCH_SIZE,
        transform=True
    )

    val_generator = DataGenerator(
        val_image_ids,
        val_pixels,
        val_labels,
        shuffle=False,
        batch_size=VAL_BATCH_SIZE,
        transform=False
    )
    print("Data generator created successfully.\n\n")
    
    # Model:
    model = YOLOv3()

    # Optimizer and Loss function:
    optimizer = Adam(LEARNING_RATE)
    loss = YOLOLoss()

    model.compile(optimizer=optimizer, loss=loss)
    print("Model compiled successfully.\n\n")

    # Checkpoint:
    os.makedirs(CHECKPOINTS_DIR, exist_ok=True)
    checkpoint_path = f"{CHECKPOINTS_DIR}/cp-{{epoch:04d}}.weights.h5"
    checkpoint_callback = ModelCheckpoint(
        filepath=checkpoint_path,
        save_weights_only=True,
        verbose=1
    )
    latest_checkpoint = tf.train.latest_checkpoint(CHECKPOINTS_DIR)
    if latest_checkpoint:
        print(f"Loading weights from the latest checkpoint: {latest_checkpoint}") 
        model.load_weights(latest_checkpoint)
        initial_epoch = int(latest_checkpoint.split("-")[-1].split(".")[0])
    else:
        print("No checkpoint found. Starting training from scratch.")
        initial_epoch = 0
    
    # Callbacks:
    callbacks = [
        ReduceLROnPlateau(monitor="loss", patience=2, verbose=1),
        EarlyStopping(monitor="loss", patience=5, verbose=1, restore_best_weights=True),
        checkpoint_callback
    ]

    # Model Training:
    history = model.fit(
        train_generator,
        validation_data=val_generator,
        epochs=EPOCHS,
        callbacks=callbacks
    )

    print("Model Trained successfully.\n")

    # Save Model:
    os.makedirs(SAVE_DIR, exist_ok=True)
    save_path = os.path.join(SAVE_DIR, MODEL_NAME)
    model.save(save_path)
    print(f"Model saved successfully at : {save_path}")

    # Status of the Model :
    print("="*50)
    print("Model Status: \n")
    print("Epochs : " + str(len(history.history["loss"])))
    print("Final Training Loss : " + str(history.history["loss"][-1]))
    print("Final Validation Loss : " + str(history.history["val_loss"][-1]))
    print("Min. Training Loss : " + str(min(history.history["loss"])))
    print("Min. Validation Loss : " + str(min(history.history["val_loss"])))
    print("="*50)
    print("\n\n")
    
    # Plot the loss curves :
    fig, ax = plt.subplots(1, 2, figsize=(15, 5))
    ax[0].set_title('Training Loss')
    ax[0].plot(history.history['loss'])
    ax[1].set_title('Validation Loss')
    ax[1].plot(history.history['val_loss'])
    plt.show()


if __name__ == "__main__":
    train()