# Global Wheat Detection using YOLOv3

## Overview

The `Global Wheat Detection using YOLOv3` project is inspired by the Kaggle competition on Global wheat detection. The primary goal of this project is to detect wheat heads in images by predicting bounding boxes around them. For this purpose, a custom object detection model has been developed that closely resembles `YOLOv3`, an advanced deep-learning architecture for real-time object detection. This model is capable of analyzing input images, identifying wheat heads, and returning the processed images with bounding boxes drawn around detected areas.

The project follows a modularized architecture, ensuring clean, scalable, and maintainable code. It includes dedicated components for data ingestion, data cleaning, data generation, model training, inference, and utility functions, all structured within the `src` directory. 

To make the model easily accessible, the project has been deployed as a Flask web application. Users can upload an image through the web interface, and the model will process the image and return the output with detected wheat heads. To further improve accessibility and deployment efficiency, the entire application has been containerized using Docker, allowing seamless execution across different environments without dependency issues.

## Table of Contents 
- [Overview](#overview)
- [Features](#features)
- [Project Structure](#project-structure)
- [Dataset](#dataset)
- [Installation](#installation)
  - Clone the Repository
  - Set Up a Virtual Environment
  - Install Dependencies
- [Model Details](#model-details)
- [Model Training](#model-training)
- [Model Prediction](#model-prediction)
- [Training and Prediction Workflow](#training-and-prediction-workflow)
- [Docker Deployment](#docker-deployment)  
  - Build the Docker Image
  - Run the Docker Container
  - Access the Application
- [Live Demo](https://www.youtube.com/watch?v=p7uqx4rg4ME)
- [Contributions](#contributions)
- [License](#license)

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

This is the architecture of YOLOv3 (mentioned in the original research paper), although our custom model resembles it, due to its dependency on hardware, some layers of it are modified. \
![image](https://editor.analyticsvidhya.com/uploads/8829414.png)

To refine the object detection process, **Non-Maximum Suppression (NMS)** has been implemented to filter out redundant bounding boxes and retain only the most confident detections. The model uses **Intersection over Union (IoU)** as a metric to measure the overlap between predicted and ground-truth bounding boxes. This ensures that the final bounding boxes accurately represent distinct wheat heads, preventing excessive overlapping detections. These techniques significantly enhance the precision of the model by eliminating false positives and improving localization accuracy. The techniques have been implemented in `utils.py` file.

## Model Training
The training process for the Global Wheat Detection using YOLOv3 project is structured using a dedicated pipeline file, `training.py`, ensuring a clean and modular approach. The pipeline follows a well-defined sequence of steps, including data ingestion, preprocessing, model training, and evaluation.

**1. Setting Hyperparameters :**
Before training begins, key hyperparameters are defined in training.py to control the model’s learning process:
```
TRAIN_BATCH_SIZE = 10  
VAL_BATCH_SIZE = 5  
EPOCHS = 10  
LEARNING_RATE = 1e-4  
CHECKPOINTS_DIR = "./checkpoints"  
SAVE_DIR = "./models"  
MODEL_NAME = "yolo_model.h5"
```
These hyperparameters dictate the batch size for training and validation, the total number of epochs, learning rate, and model checkpointing locations.

**2. Data Ingestion and Cleaning :**
To handle data efficiently, two custom modules, data_ingestion.py and data_cleaning.py, are used. The data ingestion module loads image arrays and corresponding bounding box (bbox) data from preprocessed artifacts. The data cleaning module filters out irrelevant bounding boxes, ensuring that only valid annotations are used for training.

**3. Custom Data Generator :**
Instead of using standard data loaders, a custom data generator is created to feed images and annotations into the model. This approach enhances flexibility in data augmentation, batch loading, and preprocessing (includes image resizing, converting bounding boxes to YOLO format, saving image, label as batches, etc.). The generator ensures that the model receives properly formatted image-bbox pairs during training.

**4. Model Definition and Compilation :**
The model architecture, resembling YOLOv3, is implemented in model.py. It is designed to detect wheat heads in images with high accuracy. The model is compiled with the following settings:

**5. Optimizer: Adam (Adaptive Moment Estimation) :**
Loss Function: A custom YOLO loss inspired by the original YOLOv3 research paper, implemented in loss.py. This loss function ensures precise bounding box regression and class probability predictions.

**6. Callbacks for Efficient Training :**
To enhance model performance and prevent overfitting, multiple callbacks are used:

ModelCheckpoint: Saves the best model weights during training.
ReduceLROnPlateau: Reduces the learning rate when training stagnates.
EarlyStopping: Stops training when the validation loss stops improving.
These callbacks improve training stability and generalization.

**7. Training and Saving the Model :**
The model is trained using model.fit(), with the custom data generator supplying input images and bounding boxes. Once training is complete, the final model is saved in the models/ directory with a size of approximately 2 GiB.

**8. Logging and Exception Handling :**
The entire training process is wrapped with custom exception handling and logging mechanisms from logging.py and exception.py. These ensure smooth debugging and better error tracking.

**9. Training Performance Visualization :**
After training, a plot of training and validation loss is generated, allowing an analysis of how well the model has learned over time.

## Model Prediction
In this section, we use the pre-trained model stored in the `models/` directory to make predictions on new images. The steps involved are as follows:

**1. Set Hyperparameters :**
First, set the following hyperparameters (the same ones used during model training):
```
MODEL_NAME = "yolo_model.h5"
SAVE_DIR = "./models"
VAL_BATCH_SIZE = 5
PRED_DIR = "./predictions"
TEST_IMAGE_DIR = "./artifacts/test"
```

**2. Load the Trained Model :**
The pre-trained model is loaded from the models directory using the specified SAVE_DIR and MODEL_NAME. This model will be used for predictions:
```
model = load_model(os.path.join(SAVE_DIR, MODEL_NAME))
```

**3. Preprocess Test Images :**
The images from the TEST_IMAGE_DIR directory are preprocessed in the same manner as during training. This includes resizing, normalization, and any other transformations applied in the training pipeline.

**4. Run Predictions on Test Images :**
After preprocessing, the model is used to make predictions on the test images. These predictions typically involve detecting objects and outputting bounding boxes and class labels.

**5. Save Annotated Images :**
Once the predictions are made, the results (including bounding boxes and labels) are annotated on the images. These annotated images are saved in the PRED_DIR directory for later review.

**6. Directory Structure :**
Make sure the following directory structure is set up:
```
./models: Contains the saved model file (yolo_model.h5).
./predictions: The directory where annotated images will be saved.
./artifacts/test: The directory where the test images are located.
```

## Training and Prediction Workflow
Follow the steps below to train the model, make predictions on test images, and run the Flask app for visualizing the results:

**1. Training the Model :**
To train the model, ensure that you have a GPU available on your device for efficient training. Once ready, run the following command:
```
python -m src.pipeline.training
```
This will start the training process and save the trained model for later use.

**2. Making Predictions on Test Images (Optional but Recommended) :**
After training the model, you can test it by running predictions on the test images. This step is optional but highly recommended to evaluate the performance. Run the following command:
```
python -m src.pipeline.prediction
```
The model will process the test images, and the annotated images will be saved in the ./predictions directory.

**3. Running the Flask App :**
To run the Flask application and check the results in a web interface, use the following command:
```
python app.py
```
You can then view the results by navigating to http://localhost:5000 in your browser.

## Docker Deployment
Deploying the model and Flask application using Docker ensures a consistent environment and ease of deployment across different platforms. Follow these steps to containerize and deploy your application.

**1. Build the Docker Image :**
First, ensure that your project directory contains a `Dockerfile` (same as in the repository) that defines the steps to set up the environment for your application. 
Now, build the Docker image by running the following command in your project directory:
```
docker build -t wheat-detection-app .
```
This will create a Docker image named `wheat-detection-app` based on the instructions in the Dockerfile.

**2. Run the Docker Container :**
Once the image is built, you can run the container with the following command:
```
docker run -p 5000:5000 wheat-detection-app
```

**3. Access the Application :**
After running the Docker container, the Flask app will be accessible at http://localhost:5000 in your web browser. This URL will serve the app as you interact with it through the Flask interface.

## Contributions
We welcome contributions to this project! If you'd like to contribute, please follow these steps:

- Fork the repository and create a new branch.
- Make your changes and ensure that your code follows the project's style and guidelines.
- Test your changes to make sure everything works as expected.
- Submit a pull request with a description of the changes you've made.

Your contributions help improve the project, and we appreciate your effort!

## License
This project is licensed under the MIT License - see the [LICENSE](https://github.com/LinconDash/Global-Wheat-Detection-YOLOv3/blob/master/LICENSE) file for details.
