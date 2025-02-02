# Global Wheat Detection using YOLOv3

## Overview

The `Global Wheat Detection using YOLOv3` project is inspired by the Kaggle competition on Global wheat detection. The primary goal of this project is to detect wheat heads in images by predicting bounding boxes around them. For this purpose, a custom object detection model has been developed that closely resembles `YOLOv3`, an advanced deep-learning architecture for real-time object detection. This model is capable of analyzing input images, identifying wheat heads, and returning the processed images with bounding boxes drawn around detected areas.

The project follows a modularized architecture, ensuring clean, scalable, and maintainable code. It includes dedicated components for data ingestion, data cleaning, data generation, model training, inference, and utility functions, all structured within the `src` directory. 

To make the model easily accessible, the project has been deployed as a Flask web application. Users can upload an image through the web interface, and the model will process the image and return the output with detected wheat heads. To further improve accessibility and deployment efficiency, the entire application has been containerized using Docker, allowing seamless execution across different environments without dependency issues.

## Table of Contents 
- [Overview](#overview)
- Features
- Project Structure
- Dataset
- Installation
  - Clone the Repository
  - Set Up a Virtual Environment
  - Install Dependencies
- Model Details
- Model Training and Prediction 
  - Run the training pipeline
  - Run the prediction pipeline (optional)
  - Run the Flask App
- Docker Deployment  
  - Build the Docker Image
  - Run the Docker Container
  - Tutorial(optional)
- Usage
- Contributions
- License

## Features
- **Custom YOLOv3 Model and YOLO loss:** Detects wheat heads and outputs images with bounding boxes. The model and the loss are present in `model.py` and `loss.py` resembles that of mentioned in the original YOLO paper by the researchers
- **Modularized Codebase:** Structured code with components and pipelines is found easier to be scalable by the developers.
- **Flask Web App Deployment:** Provides an interface to upload images containing wheat heads and get predictions.
- **Docker Containerization:** Simplifies deployment and ensures reproducibility.
- **Cloud Deployment:** In future, this project can be deployed in cloud using some minor modifications as it is already containerized.

## Project Structure
```
Global-Wheat-Detection-YOLOv3/
|-- artifacts/
|   |-- test/                  # contains the test images  
|   |-- train/                 # contains the training images
|   └── train.csv              # contains the bounding box info about the training images (COCO format)
|-- notebook/
|   |-- Exploratory-data-analysis.ipynb
|   └── Model-evaluation.ipynb
| -- src/
|   │-- components/            # contains components such as Yolov3 model, Yolo loss, data cleaning, data generator, etc.
|   │   │-- data_ingestion.py  
|   |   │-- data_cleaning.py
|   │   │-- data_generator.py
|   │   │-- model.py
|   │   └── loss.py
|   |-- pipeline/
|   |    |--prediction.py
|   |    └── training.py
|   |-- exception.py            # contains custom exception handling   
|   |--logger.py                # contains logging methodology
|   └──utils.py                 # contains important utility functions
|
│-- static/                     # contains css and js required for web app 
|   |--css        
│   └──js
│-- templates/                  # contains html required for web app
│   └── index.html
│-- app.py                      # Flask application 
│-- Dockerfile                  # Docker file for containerization
│-- requirements.txt            # Dependencies and modules required
|-- .gitignore
|-- .dockerignore
└── setup.py
```
Note : In addition to these , several other files and folders are created during the execution of some files.

## Dataset
- The data used for this project belongs to one of the infamous kaggle competition called `Global wheat detection`. 

- There are 2 separate directories called `test` and `train` which contains images of wheat head and another csv file called `train.csv` which contains the data about the bounding boxes of the training images
with their corresponding image_ids and the bounding boxes are in the COCO format representation which is converted to YOLO format later while creating data generators.    

- You can find more details on the provided [link](https://www.kaggle.com/competitions/global-wheat-detection) 

## Installation
To set up the project locally, follow these steps:

**1. Clone the Repository:**
  - This is the most crucial step in the installation
  - Use these commands to clone the repository to your local repository for further execution
```
git clone https://github.com/LinconDash/Global-Wheat-Detection-YOLOv3.git
cd Global-Wheat-Detection-YOLOv3
```

**2. Set Up a Virtual Environment (Optional but Recommended):**
  - Although it is not compulsory but it is a best practice to crreate a virtual environment inorder to avoid dependencies conflict issue
  - Create a virual environment `venv` and activate it using these commands 
```
python -m venv venv
source venv/bin/activate   # On MacOS/Linux
# or
venv\Scripts\activate      # On Windows
```

**3. Install Dependencies:**
  - There are 2 ways to install the dependencies (modules) that are required for the project
```
pip install -r requirements.txt
# or
pip install .       # (only works if you have setup.py)
```

## Model Details 
The model implemented in this project is based on the `YOLOv3` (You Only Look Once v3) architecture, a widely used deep-learning algorithm for real-time object detection. It has been modified and fine-tuned specifically for the task of wheat head detection. The model utilizes a custom dataset pipeline made up of data genrators for loading, cleaning, preprocessing and manipulating the data, ensuring optimized input for training. Various data augmentation techniques from the albumentations have been applied to enhance model generalization and robustness. The model is trained using a custom YOLO loss (bounding box regression loss) function that is based on the original research paper, which allows it to accurately predict the location of wheat heads in images by drawing the bouding box around it. By leveraging Convolutional Neural Networks (CNNs) and a multi-scale feature detection approach, the model effectively identifies wheat heads in complex agricultural images, providing high accuracy and reliable detections. The architecture is powerful and computationally heavy, making it well-suited for devices with gpu(s) to train and deploy in real-world scenarios, including web applications.
After the model is trained a new directory called `models` is created where the model is stored with its entire architecture for future use or fine-tuning.
