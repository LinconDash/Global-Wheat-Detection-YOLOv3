import io
import sys
import numpy as np
import tensorflow as tf
import albumentations as albu

from flask import Flask, request, send_file, render_template
from PIL import Image, ImageDraw
from src.utils import non_max_suppression, predictions_to_bboxes
from src.components.data_generator import DataGenerator
from src.exception import CustomException
from src.logger import logging

app = Flask(__name__)

# Load your model
logging.info("Loading the trained model")
MODEL_PATH = "models/yolo_model.h5"
model = tf.keras.models.load_model(MODEL_PATH)
print("Model loaded successfully.")


# Function to draw bounding boxes
def draw_bboxes(image, bboxes):
    draw = ImageDraw.Draw(image)
    for bbox in bboxes:
        x, y, w, h = bbox
        draw.rectangle([x, y, x+w, y+h], outline="red", width=2)
        draw.circle([(x+w/2) , (y+h/2)], radius=1, width=2, outline="red")
    return image

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        if 'file' not in request.files:
            return "No file uploaded", 400

        file = request.files['file']
        image = Image.open(file.stream).convert("RGB")
        image_resized = image.resize((256, 256))  

        # Convert image to numpy array for model inference
        pixels = np.asarray(image_resized)
        
        val_augmentations = albu.Compose([
            albu.CLAHE(p=1),
            albu.ToGray(p=1)
        ])

        aug_result = val_augmentations(image=pixels)
        pixels = np.array(aug_result['image']) / 255
        pixels = np.expand_dims(pixels, axis=0)

        # Predict bounding boxes
        bboxes = model.predict(pixels)
        image_grid = DataGenerator().image_grid
        bboxes = np.squeeze(bboxes)

        predictions = predictions_to_bboxes(bboxes, image_grid)
        filtered_bboxes = non_max_suppression(predictions, iou_threshold=0.3, prob_threshold=0.3)
        print(filtered_bboxes.shape)
        filtered_bboxes[:, 2:4] = filtered_bboxes[:,2:4] - filtered_bboxes[:,0:2]

        # Draw bounding boxes
        result_image = draw_bboxes(image_resized, filtered_bboxes)

        # Convert image to bytes for response
        img_io = io.BytesIO()
        result_image.save(img_io, 'JPEG')
        img_io.seek(0)
        return send_file(img_io, mimetype='image/jpeg')
    
    except Exception as e:
        raise CustomException(e, sys)

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=5000, debug=False)