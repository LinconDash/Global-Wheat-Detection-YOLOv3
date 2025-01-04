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

def predictions_to_bboxes(bboxes, image_grid):
    """
        Since we converted the image into 32X32 grid 
        we need to convert the predictions to bboxes for entire image for vizualization
        Converting from [x, y, w, h] to [x_min, y_min, x_max, y_max] along with the entire image dimension. (0 to 255)
    """
    bboxes = bboxes.copy()
    
    im_width = (image_grid[:,:,2] * 32)
    im_height = (image_grid[:,:,3] * 32)
    
    # descale x,y
    bboxes[:,:,1] = (bboxes[:,:,1] * image_grid[:,:,2]) + image_grid[:,:,0]
    bboxes[:,:,2] = (bboxes[:,:,2] * image_grid[:,:,3]) + image_grid[:,:,1]
    bboxes[:,:,6] = (bboxes[:,:,6] * image_grid[:,:,2]) + image_grid[:,:,0]
    bboxes[:,:,7] = (bboxes[:,:,7] * image_grid[:,:,3]) + image_grid[:,:,1]
    
    # descale width,height
    bboxes[:,:,3] = bboxes[:,:,3] * im_width 
    bboxes[:,:,4] = bboxes[:,:,4] * im_height
    bboxes[:,:,8] = bboxes[:,:,8] * im_width 
    bboxes[:,:,9] = bboxes[:,:,9] * im_height
    
    # centre x,y to top left x,y
    bboxes[:,:,1] = bboxes[:,:,1] - (bboxes[:,:,3] / 2)
    bboxes[:,:,2] = bboxes[:,:,2] - (bboxes[:,:,4] / 2)
    bboxes[:,:,6] = bboxes[:,:,6] - (bboxes[:,:,8] / 2)
    bboxes[:,:,7] = bboxes[:,:,7] - (bboxes[:,:,9] / 2)
    
    # width,heigth to x_max,y_max
    bboxes[:,:,3] = bboxes[:,:,1] + bboxes[:,:,3]
    bboxes[:,:,4] = bboxes[:,:,2] + bboxes[:,:,4]
    bboxes[:,:,8] = bboxes[:,:,6] + bboxes[:,:,8]
    bboxes[:,:,9] = bboxes[:,:,7] + bboxes[:,:,9]
    
    return bboxes

def intersection_over_union(prediction, target):
    """
        Returns the intersection over union area that is used for non max suppression
        It returns for one anchor box only
    """
    box1_x1 = prediction[..., 0]
    box1_y1 = prediction[..., 1]
    box1_x2 = prediction[..., 2]
    box1_y2 = prediction[..., 3]  
    box2_x1 = target[..., 0]
    box2_y1 = target[..., 1]
    box2_x2 = target[..., 2]
    box2_y2 = target[..., 3]

    x1 = max(box1_x1, box2_x1)
    x2 = min(box1_x2, box2_x2)
    y1= max(box1_y1, box2_y1)
    y2 = min(box1_y2, box2_y2)

    intersection = (x2 - x1) * (y2 - y1)

    box1_area = abs((box1_x2 - box1_x1) * (box1_y2 - box1_y1))
    box2_area = abs((box2_x2 - box2_x1) * (box2_y2 - box2_y1))
    
    return intersection / (box1_area + box2_area - intersection + 1e-6)

def non_max_suppression(bboxes, iou_threshold, prob_threshold):
    bboxes = bboxes.reshape(-1, 10)
    
    # Combine anchor box probabilities and filter by the threshold
    combined_bboxes = []
    for box in bboxes:
        if box[0] > prob_threshold or box[5] > prob_threshold:
            # Select the anchor with the higher probability
            if box[0] >= box[5]:
                combined_bboxes.append([box[0], *box[1:5]])  # First anchor
            else:
                combined_bboxes.append([box[5], *box[6:10]])  # Second anchor

    # Sort by object probability
    combined_bboxes = sorted(combined_bboxes, key=lambda x: x[0], reverse=True)
    combined_bboxes = np.array(combined_bboxes)
    bboxes_after_nms = []

    # Apply NMS
    while len(combined_bboxes) > 0:
        chosen_box = combined_bboxes[0]  # Select the box with the highest probability
        bboxes_after_nms.append(chosen_box)
        combined_bboxes = combined_bboxes[1:]  # Remove the chosen box
        combined_bboxes = [
            box for box in combined_bboxes
            if intersection_over_union(chosen_box[1:5], box[1:5]) < iou_threshold
        ]
    if bboxes_after_nms == []:
        return np.zeros((1, 4))
    return np.array(bboxes_after_nms)[..., 1:5]