import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf

from tqdm.auto import tqdm
from PIL import Image, ImageDraw, ImageEnhance
from src.components.model import YOLOv3
from src.exception import CustomException

def str_to_float(group):
    try:
        bboxes = group["bbox"].str.split(",", expand=True)
        bboxes[0] = bboxes[0].str.slice(start=1)
        bboxes[3] = bboxes[3].str.slice(stop=-1)
        return bboxes.values.astype(float)
    except Exception as e:
        raise CustomException(e, sys)

def load_image_as_pixels(file_path:str, image_id:str):
    try:
        image = Image.open(file_path + "/" + image_id + ".jpg")
        image = image.resize((256, 256))
        return np.asarray(image)
    except Exception as e:
        raise CustomException(e, sys)

def plot_bboxes(image_id:str, bboxes, data="train"):
    try:
        image = Image.open(data + "/" + image_id + ".jpg")
        image = image.resize((256, 256)) 
        draw = ImageDraw.Draw(image)
        for bbox in bboxes:
            draw_bbox(draw, bbox)
        return np.asarray(image) / 255
    except Exception as e:
        raise CustomException(e, sys)

def draw_bbox(draw, bbox):
    try:
        x, y, w, h = bbox
        draw.rectangle([x, y, x+w, y+h], width=2, outline="cyan")
        draw.circle([(x+w/2) , (y+h/2) ], radius=1, width=2, outline="cyan")
        return
    except Exception as e:
        raise CustomException(e, sys)

def plot_images_with_bboxes(image_ids, image_labels, data="train"):
    try:
        img_pixels = []
        for img_id in np.unique(image_ids):
            bboxes_of_img = image_labels[img_id]
            img_pixels.append(plot_bboxes(img_id, bboxes_of_img, data))

        n = len(image_ids)
        if n == 1:
            plt.imshow(img_pixels[0])
            return
            
        fig, axes = plt.subplots(1, n, figsize=(20*n, 20*n))
        for i, img_pixel in enumerate(img_pixels):
            axes[i].imshow(img_pixel)
        return

    except Exception as e:
        raise CustomException(e, sys)

def show_images(n:int):
    try:
        labels = pd.read_csv("artifacts/train.csv")
        labels = labels.groupby("image_id").apply(str_to_float)
        all_image_ids = list(labels.index.values)
        image_ids = random.sample(all_image_ids, k=n)
        image_pixels = []
        
        for img_id in image_ids:
            if img_id in train_labels:
                bboxes_of_img = train_labels[img_id]
            else:
                bboxes_of_img = val_labels[img_id]
            image_pixels.append(plot_bboxes(img_id, bboxes_of_img))
        
        fig, axes = plt.subplots(1, n, figsize=(20*n, 20*n))
        for i, img_pixel in enumerate(image_pixels):
            axes[i].imshow(img_pixel)
    except Exception as e:
        raise CustomException(e, sys)

def load_model(save_dir, filename="yolov3_model.h5"):
    load_path = os.path.join(save_dir, filename)
    if not os.path.exists(load_path):
        raise FileNotFoundError(f"No such model file exists : {load_path}")
    
    model = tf.keras.models.load_model(
        load_path,
        custom_objects={"YOLOv3":YOLOv3}
    )
    print(f"Model loaded successfully from : {load_path}")
    return model

def save_model(model, save_dir, filename="yolov3_model.h5"):
    try:
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, filename)
        model.save(save_path)
        print(f"Model saved successfully at : {save_path}")
    except Exception as e:
        raise CustomException(e, sys)