'''
Note : This Model architecture only resembles the original YOLOv3 architecture ,
       and contains much less complexity and layers than the original YOLOv3 Model.
       Also some segments may seem repeated but that is how a class Model works for 
       effectively identifying the model's trainable and non trainable parameters and 
       also for improving the readability of code. 
'''

import os
import sys
import warnings
import tensorflow as tf

from tensorflow import keras
from tensorflow.keras import Model
from tensorflow.keras.layers import (
    Dense, 
    Add,
    Flatten,
    Conv2D,
    LeakyReLU,
    BatchNormalization
)

from src.logger import logging
from src.exception import CustomException
warnings.filterwarnings("ignore")

class YOLOv3(Model):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        logging.info("Creating YOLOv3 Model.")

        # Initial Convolutional layer:
        self.conv1 = Conv2D(32, (3, 3), strides=(1, 1), padding="same") 
        self.bn1 = BatchNormalization()
        self.act1 = LeakyReLU(0.1)

        # Five block configuration:
        
        # Block 1:
        self.block1_conv1 = Conv2D(64, (3, 3), strides=(1, 1), padding="same") 
        self.block1_bn1 = BatchNormalization()
        self.block1_act1 = LeakyReLU(0.1)

        self.block1_conv2_1 = Conv2D(32, (3, 3), strides=(1, 1), padding="same") 
        self.block1_bn2_1 = BatchNormalization()
        self.block1_act2_1 = LeakyReLU(0.1)

        self.block1_conv3_1 = Conv2D(64, (3, 3), strides=(1, 1), padding="same") 
        self.block1_bn3_1 = BatchNormalization()
        self.block1_act3_1 = LeakyReLU(0.1)

        self.block1_add_1 = Add()
        self.block1_act4_1 = LeakyReLU(0.1)

        self.block1_conv2_2 = Conv2D(32, (3, 3), strides=(1, 1), padding="same") 
        self.block1_bn2_2 = BatchNormalization()
        self.block1_act2_2 = LeakyReLU(0.1)

        self.block1_conv3_2 = Conv2D(64, (3, 3), strides=(1, 1), padding="same") 
        self.block1_bn3_2 = BatchNormalization()
        self.block1_act3_2 = LeakyReLU(0.1)

        self.block1_add_2 = Add()
        self.block1_act4_2 = LeakyReLU(0.1)

        # Block 2:
        self.block2_conv1 = Conv2D(128, (3, 3), strides=(1, 1), padding="same") 
        self.block2_bn1 = BatchNormalization()
        self.block2_act1 = LeakyReLU(0.1)

        self.block2_conv2_1 = Conv2D(64, (3, 3), strides=(1, 1), padding="same") 
        self.block2_bn2_1 = BatchNormalization()
        self.block2_act2_1 = LeakyReLU(0.1)

        self.block2_conv3_1 = Conv2D(128, (3, 3), strides=(1, 1), padding="same") 
        self.block2_bn3_1 = BatchNormalization()
        self.block2_act3_1 = LeakyReLU(0.1)

        self.block2_add_1 = Add()
        self.block2_act4_1 = LeakyReLU(0.1)

        self.block2_conv2_2 = Conv2D(64, (3, 3), strides=(1, 1), padding="same") 
        self.block2_bn2_2 = BatchNormalization()
        self.block2_act2_2 = LeakyReLU(0.1)

        self.block2_conv3_2 = Conv2D(128, (3, 3), strides=(1, 1), padding="same") 
        self.block2_bn3_2 = BatchNormalization()
        self.block2_act3_2 = LeakyReLU(0.1)

        self.block2_add_2 = Add()
        self.block2_act4_2 = LeakyReLU(0.1)

        # Block 3:
        self.block3_conv1 = Conv2D(256, (3, 3), strides=(2, 2), padding="same") 
        self.block3_bn1 = BatchNormalization()
        self.block3_act1 = LeakyReLU(0.1)

        self.block3_conv2_1 = Conv2D(128, (3, 3), strides=(1, 1), padding="same") 
        self.block3_bn2_1 = BatchNormalization()
        self.block3_act2_1 = LeakyReLU(0.1)

        self.block3_conv3_1 = Conv2D(256, (3, 3), strides=(1, 1), padding="same") 
        self.block3_bn3_1 = BatchNormalization()
        self.block3_act3_1 = LeakyReLU(0.1)

        self.block3_add_1 = Add()
        self.block3_act4_1 = LeakyReLU(0.1)

        self.block3_conv2_2 = Conv2D(128, (3, 3), strides=(1, 1), padding="same") 
        self.block3_bn2_2 = BatchNormalization()
        self.block3_act2_2 = LeakyReLU(0.1)

        self.block3_conv3_2 = Conv2D(256, (3, 3), strides=(1, 1), padding="same") 
        self.block3_bn3_2 = BatchNormalization()
        self.block3_act3_2 = LeakyReLU(0.1)

        self.block3_add_2 = Add()
        self.block3_act4_2 = LeakyReLU(0.1)

        self.block3_conv2_3 = Conv2D(128, (3, 3), strides=(1, 1), padding="same") 
        self.block3_bn2_3 = BatchNormalization()
        self.block3_act2_3 = LeakyReLU(0.1)

        self.block3_conv3_3 = Conv2D(256, (3, 3), strides=(1, 1), padding="same") 
        self.block3_bn3_3 = BatchNormalization()
        self.block3_act3_3 = LeakyReLU(0.1)

        self.block3_add_3 = Add()
        self.block3_act4_3 = LeakyReLU(0.1)

        self.block3_conv2_4 = Conv2D(128, (3, 3), strides=(1, 1), padding="same") 
        self.block3_bn2_4 = BatchNormalization()
        self.block3_act2_4 = LeakyReLU(0.1)

        self.block3_conv3_4 = Conv2D(256, (3, 3), strides=(1, 1), padding="same") 
        self.block3_bn3_4 = BatchNormalization()
        self.block3_act3_4 = LeakyReLU(0.1)

        self.block3_add_4 = Add()
        self.block3_act4_4 = LeakyReLU(0.1)

        self.block3_conv2_5 = Conv2D(128, (3, 3), strides=(1, 1), padding="same") 
        self.block3_bn2_5 = BatchNormalization()
        self.block3_act2_5 = LeakyReLU(0.1)

        self.block3_conv3_5 = Conv2D(256, (3, 3), strides=(1, 1), padding="same") 
        self.block3_bn3_5 = BatchNormalization()
        self.block3_act3_5 = LeakyReLU(0.1)

        self.block3_add_5 = Add()
        self.block3_act4_5 = LeakyReLU(0.1)

        self.block3_conv2_6 = Conv2D(128, (3, 3), strides=(1, 1), padding="same") 
        self.block3_bn2_6 = BatchNormalization()
        self.block3_act2_6 = LeakyReLU(0.1)

        self.block3_conv3_6 = Conv2D(256, (3, 3), strides=(1, 1), padding="same") 
        self.block3_bn3_6 = BatchNormalization()
        self.block3_act3_6 = LeakyReLU(0.1)

        self.block3_add_6 = Add()
        self.block3_act4_6 = LeakyReLU(0.1)

        self.block3_conv2_7 = Conv2D(128, (3, 3), strides=(1, 1), padding="same") 
        self.block3_bn2_7 = BatchNormalization()
        self.block3_act2_7 = LeakyReLU(0.1)

        self.block3_conv3_7 = Conv2D(256, (3, 3), strides=(1, 1), padding="same") 
        self.block3_bn3_7 = BatchNormalization()
        self.block3_act3_7 = LeakyReLU(0.1)

        self.block3_add_7 = Add()
        self.block3_act4_7 = LeakyReLU(0.1)

        self.block3_conv2_8 = Conv2D(128, (3, 3), strides=(1, 1), padding="same") 
        self.block3_bn2_8 = BatchNormalization()
        self.block3_act2_8 = LeakyReLU(0.1)

        self.block3_conv3_8 = Conv2D(256, (3, 3), strides=(1, 1), padding="same") 
        self.block3_bn3_8 = BatchNormalization()
        self.block3_act3_8 = LeakyReLU(0.1)

        self.block3_add_8 = Add()
        self.block3_act4_8 = LeakyReLU(0.1)

        # Block 4:
        self.block4_conv1 = Conv2D(512, (3, 3), strides=(2, 2), padding="same") 
        self.block4_bn1 = BatchNormalization()
        self.block4_act1 = LeakyReLU(0.1)

        self.block4_conv2_1 = Conv2D(256, (3, 3), strides=(1, 1), padding="same") 
        self.block4_bn2_1 = BatchNormalization()
        self.block4_act2_1 = LeakyReLU(0.1)

        self.block4_conv3_1 = Conv2D(512, (3, 3), strides=(1, 1), padding="same") 
        self.block4_bn3_1 = BatchNormalization()
        self.block4_act3_1 = LeakyReLU(0.1)

        self.block4_add_1 = Add()
        self.block4_act4_1 = LeakyReLU(0.1)

        self.block4_conv2_2 = Conv2D(256, (3, 3), strides=(1, 1), padding="same") 
        self.block4_bn2_2 = BatchNormalization()
        self.block4_act2_2 = LeakyReLU(0.1)

        self.block4_conv3_2 = Conv2D(512, (3, 3), strides=(1, 1), padding="same") 
        self.block4_bn3_2 = BatchNormalization()
        self.block4_act3_2 = LeakyReLU(0.1)

        self.block4_add_2 = Add()
        self.block4_act4_2 = LeakyReLU(0.1)

        self.block4_conv2_3 = Conv2D(256, (3, 3), strides=(1, 1), padding="same") 
        self.block4_bn2_3 = BatchNormalization()
        self.block4_act2_3 = LeakyReLU(0.1)

        self.block4_conv3_3 = Conv2D(512, (3, 3), strides=(1, 1), padding="same") 
        self.block4_bn3_3 = BatchNormalization()
        self.block4_act3_3 = LeakyReLU(0.1)

        self.block4_add_3 = Add()
        self.block4_act4_3 = LeakyReLU(0.1)

        self.block4_conv2_4 = Conv2D(256, (3, 3), strides=(1, 1), padding="same") 
        self.block4_bn2_4 = BatchNormalization()
        self.block4_act2_4 = LeakyReLU(0.1)

        self.block4_conv3_4 = Conv2D(512, (3, 3), strides=(1, 1), padding="same") 
        self.block4_bn3_4 = BatchNormalization()
        self.block4_act3_4 = LeakyReLU(0.1)

        self.block4_add_4 = Add()
        self.block4_act4_4 = LeakyReLU(0.1)

        self.block4_conv2_5 = Conv2D(256, (3, 3), strides=(1, 1), padding="same") 
        self.block4_bn2_5 = BatchNormalization()
        self.block4_act2_5 = LeakyReLU(0.1)

        self.block4_conv3_5 = Conv2D(512, (3, 3), strides=(1, 1), padding="same") 
        self.block4_bn3_5 = BatchNormalization()
        self.block4_act3_5 = LeakyReLU(0.1)

        self.block4_add_5 = Add()
        self.block4_act4_5 = LeakyReLU(0.1)

        self.block4_conv2_6 = Conv2D(256, (3, 3), strides=(1, 1), padding="same") 
        self.block4_bn2_6 = BatchNormalization()
        self.block4_act2_6 = LeakyReLU(0.1)

        self.block4_conv3_6 = Conv2D(512, (3, 3), strides=(1, 1), padding="same") 
        self.block4_bn3_6 = BatchNormalization()
        self.block4_act3_6 = LeakyReLU(0.1)

        self.block4_add_6 = Add()
        self.block4_act4_6 = LeakyReLU(0.1)

        self.block4_conv2_7 = Conv2D(256, (3, 3), strides=(1, 1), padding="same") 
        self.block4_bn2_7 = BatchNormalization()
        self.block4_act2_7 = LeakyReLU(0.1)

        self.block4_conv3_7 = Conv2D(512, (3, 3), strides=(1, 1), padding="same") 
        self.block4_bn3_7 = BatchNormalization()
        self.block4_act3_7 = LeakyReLU(0.1)

        self.block4_add_7 = Add()
        self.block4_act4_7 = LeakyReLU(0.1)

        self.block4_conv2_8 = Conv2D(256, (3, 3), strides=(1, 1), padding="same") 
        self.block4_bn2_8 = BatchNormalization()
        self.block4_act2_8 = LeakyReLU(0.1)

        self.block4_conv3_8 = Conv2D(512, (3, 3), strides=(1, 1), padding="same") 
        self.block4_bn3_8 = BatchNormalization()
        self.block4_act3_8 = LeakyReLU(0.1)

        self.block4_add_8 = Add()
        self.block4_act4_8 = LeakyReLU(0.1)

        # Block 5:
        self.block5_conv1 = Conv2D(1024, (3, 3), strides=(2, 2), padding="same") 
        self.block5_bn1 = BatchNormalization()
        self.block5_act1 = LeakyReLU(0.1)

        self.block5_conv2_1 = Conv2D(512, (3, 3), strides=(1, 1), padding="same") 
        self.block5_bn2_1 = BatchNormalization()
        self.block5_act2_1 = LeakyReLU(0.1)

        self.block5_conv3_1 = Conv2D(1024, (3, 3), strides=(1, 1), padding="same") 
        self.block5_bn3_1 = BatchNormalization()
        self.block5_act3_1 = LeakyReLU(0.1)

        self.block5_add_1 = Add()
        self.block5_act4_1 = LeakyReLU(0.1)

        self.block5_conv2_2 = Conv2D(512, (3, 3), strides=(1, 1), padding="same") 
        self.block5_bn2_2 = BatchNormalization()
        self.block5_act2_2 = LeakyReLU(0.1)

        self.block5_conv3_2 = Conv2D(1024, (3, 3), strides=(1, 1), padding="same") 
        self.block5_bn3_2 = BatchNormalization()
        self.block5_act3_2 = LeakyReLU(0.1)

        self.block5_add_2 = Add()
        self.block5_act4_2 = LeakyReLU(0.1)

        self.block5_conv2_3 = Conv2D(512, (3, 3), strides=(1, 1), padding="same") 
        self.block5_bn2_3 = BatchNormalization()
        self.block5_act2_3 = LeakyReLU(0.1)

        self.block5_conv3_3 = Conv2D(1024, (3, 3), strides=(1, 1), padding="same") 
        self.block5_bn3_3 = BatchNormalization()
        self.block5_act3_3 = LeakyReLU(0.1)

        self.block5_add_3 = Add()
        self.block5_act4_3 = LeakyReLU(0.1)

        self.block5_conv2_4 = Conv2D(512, (3, 3), strides=(1, 1), padding="same") 
        self.block5_bn2_4 = BatchNormalization()
        self.block5_act2_4 = LeakyReLU(0.1)

        self.block5_conv3_4 = Conv2D(1024, (3, 3), strides=(1, 1), padding="same") 
        self.block5_bn3_4 = BatchNormalization()
        self.block5_act3_4 = LeakyReLU(0.1)

        self.block5_add_4 = Add()
        self.block5_act4_4 = LeakyReLU(0.1)

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
        try:
            x = inputs
            x = self.conv1(x)
            x = self.bn1(x)
            x = self.act1(x)

            # Block 1 :
            x = self.block1_conv1(x)
            x = self.block1_bn1(x)
            x = self.block1_act1(x)
            x_shortcut = x
            
            x = self.block1_conv2_1(x)
            x = self.block1_bn2_1(x)
            x = self.block1_act2_1(x)

            x = self.block1_conv3_1(x)
            x = self.block1_bn3_1(x)
            x = self.block1_act3_1(x)
            x = self.block1_add_1([x_shortcut, x])
            x = self.block1_act4_1(x)
            x_shortcut = x

            x = self.block1_conv2_2(x)
            x = self.block1_bn2_2(x)
            x = self.block1_act2_2(x)

            x = self.block1_conv3_2(x)
            x = self.block1_bn3_2(x)
            x = self.block1_act3_2(x)
            x = self.block1_add_2([x_shortcut, x])
            x = self.block1_act4_2(x)

            # Block2:
            x = self.block2_conv1(x)
            x = self.block2_bn1(x)
            x = self.block2_act1(x)
            x_shortcut = x

            x = self.block2_conv2_1(x)
            x = self.block2_bn2_1(x)
            x = self.block2_act2_1(x)

            x = self.block2_conv3_1(x)
            x = self.block2_bn3_1(x)
            x = self.block2_act3_1(x)
            x = self.block2_add_1([x_shortcut, x])
            x = self.block2_act4_1(x)
            x_shortcut = x

            x = self.block2_conv2_2(x)
            x = self.block2_bn2_2(x)
            x = self.block2_act2_2(x)

            x = self.block2_conv3_2(x)
            x = self.block2_bn3_2(x)
            x = self.block2_act3_2(x)
            x = self.block2_add_2([x_shortcut, x])
            x = self.block2_act4_2(x)
            
            # Block 3:
            x = self.block3_conv1(x)
            x = self.block3_bn1(x)
            x = self.block3_act1(x)
            x_shortcut = x

            
            x = self.block3_conv2_1(x)
            x = self.block3_bn2_1(x)
            x = self.block3_act2_1(x)

            x = self.block3_conv3_1(x)
            x = self.block3_bn3_1(x)
            x = self.block3_act3_1(x)
            x = self.block3_add_1([x_shortcut, x])
            x = self.block3_act4_1(x)
            x_shortcut = x

            x = self.block3_conv2_2(x)
            x = self.block3_bn2_2(x)
            x = self.block3_act2_2(x)

            x = self.block3_conv3_2(x)
            x = self.block3_bn3_2(x)
            x = self.block3_act3_2(x)
            x = self.block3_add_2([x_shortcut, x])
            x = self.block3_act4_2(x)
            x_shortcut = x

            x = self.block3_conv2_3(x)
            x = self.block3_bn2_3(x)
            x = self.block3_act2_3(x)

            x = self.block3_conv3_3(x)
            x = self.block3_bn3_3(x)
            x = self.block3_act3_3(x)
            x = self.block3_add_3([x_shortcut, x])
            x = self.block3_act4_3(x)
            x_shortcut = x

            x = self.block3_conv2_4(x)
            x = self.block3_bn2_4(x)
            x = self.block3_act2_4(x)

            x = self.block3_conv3_4(x)
            x = self.block3_bn3_4(x)
            x = self.block3_act3_4(x)
            x = self.block3_add_4([x_shortcut, x])
            x = self.block3_act4_4(x)
            x_shortcut = x

            x = self.block3_conv2_5(x)
            x = self.block3_bn2_5(x)
            x = self.block3_act2_5(x)

            x = self.block3_conv3_5(x)
            x = self.block3_bn3_5(x)
            x = self.block3_act3_5(x)
            x = self.block3_add_5([x_shortcut, x])
            x = self.block3_act4_5(x)
            x_shortcut = x

            x = self.block3_conv2_6(x)
            x = self.block3_bn2_6(x)
            x = self.block3_act2_6(x)

            x = self.block3_conv3_6(x)
            x = self.block3_bn3_6(x)
            x = self.block3_act3_6(x)
            x = self.block3_add_6([x_shortcut, x])
            x = self.block3_act4_6(x)
            x_shortcut = x

            x = self.block3_conv2_7(x)
            x = self.block3_bn2_7(x)
            x = self.block3_act2_7(x)

            x = self.block3_conv3_7(x)
            x = self.block3_bn3_7(x)
            x = self.block3_act3_7(x)
            x = self.block3_add_7([x_shortcut, x])
            x = self.block3_act4_7(x)
            x_shortcut = x

            x = self.block3_conv2_8(x)
            x = self.block3_bn2_8(x)
            x = self.block3_act2_8(x)

            x = self.block3_conv3_8(x)
            x = self.block3_bn3_8(x)
            x = self.block3_act3_8(x)
            x = self.block3_add_8([x_shortcut, x])
            x = self.block3_act4_8(x)

            # Block 4:
            x = self.block4_conv1(x)
            x = self.block4_bn1(x)
            x = self.block4_act1(x)
            x_shortcut = x

            x = self.block4_conv2_1(x)
            x = self.block4_bn2_1(x)
            x = self.block4_act2_1(x)

            x = self.block4_conv3_1(x)
            x = self.block4_bn3_1(x)
            x = self.block4_act3_1(x)
            x = self.block4_add_1([x_shortcut, x])
            x = self.block4_act4_1(x)
            x_shortcut = x

            x = self.block4_conv2_2(x)
            x = self.block4_bn2_2(x)
            x = self.block4_act2_2(x)

            x = self.block4_conv3_2(x)
            x = self.block4_bn3_2(x)
            x = self.block4_act3_2(x)
            x = self.block4_add_2([x_shortcut, x])
            x = self.block4_act4_2(x)
            x_shortcut = x

            x = self.block4_conv2_3(x)
            x = self.block4_bn2_3(x)
            x = self.block4_act2_3(x)

            x = self.block4_conv3_3(x)
            x = self.block4_bn3_3(x)
            x = self.block4_act3_3(x)
            x = self.block4_add_3([x_shortcut, x])
            x = self.block4_act4_3(x)
            x_shortcut = x

            x = self.block4_conv2_4(x)
            x = self.block4_bn2_4(x)
            x = self.block4_act2_4(x)

            x = self.block4_conv3_4(x)
            x = self.block4_bn3_4(x)
            x = self.block4_act3_4(x)
            x = self.block4_add_4([x_shortcut, x])
            x = self.block4_act4_4(x)
            x_shortcut = x

            x = self.block4_conv2_5(x)
            x = self.block4_bn2_5(x)
            x = self.block4_act2_5(x)

            x = self.block4_conv3_5(x)
            x = self.block4_bn3_5(x)
            x = self.block4_act3_5(x)
            x = self.block4_add_5([x_shortcut, x])
            x = self.block4_act4_5(x)
            x_shortcut = x

            x = self.block4_conv2_6(x)
            x = self.block4_bn2_6(x)
            x = self.block4_act2_6(x)

            x = self.block4_conv3_6(x)
            x = self.block4_bn3_6(x)
            x = self.block4_act3_6(x)
            x = self.block4_add_6([x_shortcut, x])
            x = self.block4_act4_6(x)
            x_shortcut = x

            x = self.block4_conv2_7(x)
            x = self.block4_bn2_7(x)
            x = self.block4_act2_7(x)

            x = self.block4_conv3_7(x)
            x = self.block4_bn3_7(x)
            x = self.block4_act3_7(x)
            x = self.block4_add_7([x_shortcut, x])
            x = self.block4_act4_7(x)
            x_shortcut = x

            x = self.block4_conv2_8(x)
            x = self.block4_bn2_8(x)
            x = self.block4_act2_8(x)

            x = self.block4_conv3_8(x)
            x = self.block4_bn3_8(x)
            x = self.block4_act3_8(x)
            x = self.block4_add_8([x_shortcut, x])
            x = self.block4_act4_8(x)

            # Block 5:
            x = self.block5_conv1(x)
            x = self.block5_bn1(x)
            x = self.block5_act1(x)
            x_shortcut = x

            x = self.block5_conv2_1(x)
            x = self.block5_bn2_1(x)
            x = self.block5_act2_1(x)

            x = self.block5_conv3_1(x)
            x = self.block5_bn3_1(x)
            x = self.block5_act3_1(x)
            x = self.block5_add_1([x_shortcut, x])
            x = self.block5_act4_1(x)
            x_shortcut = x

            x = self.block5_conv2_2(x)
            x = self.block5_bn2_2(x)
            x = self.block5_act2_2(x)

            x = self.block5_conv3_2(x)
            x = self.block5_bn3_2(x)
            x = self.block5_act3_2(x)
            x = self.block5_add_2([x_shortcut, x])
            x = self.block5_act4_2(x)
            x_shortcut = x

            x = self.block5_conv2_3(x)
            x = self.block5_bn2_3(x)
            x = self.block5_act2_3(x)

            x = self.block5_conv3_3(x)
            x = self.block5_bn3_3(x)
            x = self.block5_act3_3(x)
            x = self.block5_add_3([x_shortcut, x])
            x = self.block5_act4_3(x)
            x_shortcut = x

            x = self.block5_conv2_4(x)
            x = self.block5_bn2_4(x)
            x = self.block5_act2_4(x)

            x = self.block5_conv3_4(x)
            x = self.block5_bn3_4(x)
            x = self.block5_act3_4(x)
            x = self.block5_add_4([x_shortcut, x])
            x = self.block5_act4_4(x)

            # Output layers
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

        except Exception as e:
            raise CustomException(e, sys)