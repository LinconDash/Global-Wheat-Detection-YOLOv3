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

def plot_bboxes(image_id:str, bboxes, data="artifacts/train"):
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

def plot_images_with_bboxes(image_ids, image_labels, data="artifacts/train"):
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

def predictions_to_bboxes(bboxes, image_grid):
    """
        Since we converted the image into 32X32 grid 
        we need to convert the predictions to bboxes for entire image for vizualization
        Converting from [x, y, w, h] to [x_min, y_min, x_max, y_max] along with the entire image dimension. (0 to 255)
    """
    try:
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
    except Exception as e:
        raise CustomException(e, sys)


def intersection_over_union(prediction, target):
    """
    Calculate the Intersection over Union (IoU) for two bounding boxes.

    Args:
        prediction (array-like): [x1, y1, x2, y2] for the predicted bounding box.
        target (array-like): [x1, y1, x2, y2] for the target bounding box.

    Returns:
        float: IoU value.
    """
    try:
        # Extract coordinates
        box1_x1, box1_y1, box1_x2, box1_y2 = prediction
        box2_x1, box2_y1, box2_x2, box2_y2 = target

        # Calculate intersection coordinates
        x1 = max(box1_x1, box2_x1)
        y1 = max(box1_y1, box2_y1)
        x2 = min(box1_x2, box2_x2)
        y2 = min(box1_y2, box2_y2)

        # Compute intersection area
        intersection = max(0, x2 - x1) * max(0, y2 - y1)

        # Compute areas of the two boxes
        box1_area = abs((box1_x2 - box1_x1) * (box1_y2 - box1_y1))
        box2_area = abs((box2_x2 - box2_x1) * (box2_y2 - box2_y1))

        # Compute union area and IoU
        union = box1_area + box2_area - intersection
        iou = intersection / (union + 1e-6)

        return iou
    except Exception as e:
        raise CustomException(e, sys)

def non_max_suppression(bboxes, iou_threshold=0.5, prob_threshold=0.5):
    """
    Perform Non-Max Suppression (NMS) on bounding boxes.
    
    Args:
        bboxes (numpy.ndarray): Array of shape (N, 10), where N is the number of bounding boxes.
                                Each box has 10 values: [prob1, x1, y1, w1, h1, prob2, x2, y2, w2, h2].
        iou_threshold (float): IOU threshold for NMS.
        prob_threshold (float): Probability threshold to filter boxes.

    Returns:
        numpy.ndarray: Array of filtered bounding boxes after applying NMS.
    """
    try:
        # Reshape to (N, 10) and filter by prob_threshold
        bboxes = bboxes.reshape(-1, 10)
        filtered_bboxes = []

        for box in bboxes:
            if box[0] > prob_threshold:  # Anchor 1
                filtered_bboxes.append([box[0], *box[1:5]])
            if box[5] > prob_threshold:  # Anchor 2
                filtered_bboxes.append([box[5], *box[6:10]])

        if not filtered_bboxes:
            return np.empty((0, 4))  # Return empty array if no boxes pass threshold

        # Convert to numpy array and sort by probability
        filtered_bboxes = np.array(filtered_bboxes)
        filtered_bboxes = filtered_bboxes[np.argsort(-filtered_bboxes[:, 0])]

        bboxes_after_nms = []

        # Apply Non-Max Suppression
        while len(filtered_bboxes) > 0:
            chosen_box = filtered_bboxes[0]
            bboxes_after_nms.append(chosen_box)
            filtered_bboxes = filtered_bboxes[1:]
            filtered_bboxes = np.array([
                box for box in filtered_bboxes
                if intersection_over_union(chosen_box[1:], box[1:]) < iou_threshold
            ])

        return np.array(bboxes_after_nms)[:, 1:]  # Return only [x, y, w, h]
    except Exception as e:
        raise CustomException(e, sys)


def process_predictions(predictions, image_ids, image_grid):
    try:
        bboxes = {}
        for i, image_id in tqdm(enumerate(image_ids)):
            predictions[i] = predictions_to_bboxes(predictions[i], image_grid)
            bboxes[image_id] = non_max_suppression(predictions[i], iou_threshold=0.3, prob_threshold=0.3)
            # back to coco shape
            bboxes[image_id][:,2:4] = bboxes[image_id][:,2:4] - bboxes[image_id][:,0:2]
        
        return bboxes
    except Exception as e:
        raise CustomException(e, sys)

def save_predictions(predictions, image_ids, image_pixels, dir_name):
    try:
        for image_id in image_ids:
            image_array = image_pixels[image_id]
            image = Image.fromarray(image_array)
            draw = ImageDraw.Draw(image)
            bboxes = predictions.get(image_id, [])
            for bbox in bboxes:
                x1, y1, x2, y2 = bbox
                
                draw.rectangle(
                    [(x1, y1), (x2, y2)],
                    outline="red",
                    width=2
                )

            output_path = os.path.join(dir_name, f"{image_id}.png")
            image.save(output_path)
            print(f"Saved image: {output_path}")
        
        print("All images have be saved successfully.")

    except Exception as e:
        raise CustomException(e, sys)
