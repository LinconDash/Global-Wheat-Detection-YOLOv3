'''
Note : This Model architecture only resembles the original YOLOv3 architecture ,
       and contains much less complexity and layers than the original YOLOv3 Model.
'''

import os
import sys
import warnings
import tensorflow as tf

from tensorflow import keras
from tensorflow.keras import Model, Input
from tensorflow.keras.layers import Dense, Add, Flatten, Conv2D, LeakyReLU, BatchNormalization
from src.logger import logging
from src.exception import CustomException
warnings.filterwarnings("ignore")

class YOLOv3(Model):
    def __init__(self):
        super().__init__()
        logging.info("Creating YOLOv3 Model.")

        # Initial Convolutional layer:
        self.conv1 = Conv2D(32, (3, 3), strides=(1, 1), padding="same") 
        self.bn1 = BatchNormalization()
        self.act1 = LeakyReLU(0.1)

        # Five block configuration:
        self.block_configs = [
                            (64, 2, False),
                            (128, 2, False),
                            (256, 8, True),
                            (512, 8, True),
                            (1024, 4, True)
                        ]

        # Output layers:
        self.conv_out1 = Conv2D(512, (3, 3), strides=(1, 1), padding="same")
        self.bn_out1 = BatchNormalization()
        self.act_out1 = LeakyReLU(0.1)

        self.conv_out2 = Conv2D(256, (3, 3), strides=(1, 1), padding="same")
        self.bn_out2 = BatchNormalization()
        self.act_out2 = LeakyReLU(0.1)

        self.conv_out3 = Conv2D(128, (3, 3), strides=(1, 1), padding="same")
        self.bn_out3 = BatchNormalization()
        self.act_out3 = LeakyReLU(0.1)

        # Prediction layer:
        self.pred = Conv2D(10, (1, 1), strides=(1, 1), padding="same", activation="sigmoid")


    def call(self, inputs):
        x = inputs
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.act1(x)

        for units, num_repeats, downsample in self.block_configs:
            if downsample is True:
                x = Conv2D(units, (3, 3), strides=(2, 2), padding="same")(x)
            else:
                x = Conv2D(units, (3, 3), strides=(1, 1), padding="same")(x)
            x = BatchNormalization()(x)
            x = LeakyReLU(0.1)(x)
            x_shortcut = x
            
            for i in range(num_repeats):
                x = Conv2D(units // 2, (3, 3), strides=(1, 1), padding="same")(x)
                x = BatchNormalization()(x)
                x = LeakyReLU(0.1)(x)

                x = Conv2D(units, (3, 3), strides=(1, 1), padding="same")(x)
                x = BatchNormalization()(x)
                x = LeakyReLU(0.1)(x)

                x = Add()([x_shortcut, x])
                x = LeakyReLU(0.1)(x)

                x_shortcut = x

        x = self.conv_out1(x)
        x = self.bn_out1(x)
        x = self.act_out1(x)

        x = self.conv_out2(x)
        x = self.bn_out2(x)
        x = self.act_out2(x)

        x = self.conv_out3(x)
        x = self.bn_out3(x)
        x = self.act_out3(x)
        
        predictions = self.pred(x)

        return predictions

if __name__ == "__main__":
    # Model initialization
    model = YOLOv3()

    # Define input shape
    input_shape = (256, 256, 3)
    dummy_input = tf.random.uniform((1, *input_shape))  # Create dummy input

    # Build the model by calling it with the input
    dummy_output = model(dummy_input)

    # Print model summary
    print(model.summary())