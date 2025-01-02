'''
Note : This Loss function might not resemble the actual loss function in the paper,
       But it is more of a very simpler version that works with same functionality and principle
       as mentioned in the YOLO paper (except the Intersection Over Union functionality).  
'''


import sys
import tensorflow as tf

from tensorflow.keras.losses import Loss
from src.logger import logging
from src.exception import CustomException

class YOLOLoss(Loss):
    def __init__(self, name="YOLOLoss"):
        super().__init__(name=name)
        logging.info("Initializing the YOLO Loss.")
        
        self.binary_crossentropy = tf.keras.losses.BinaryCrossentropy(
            reduction = tf.keras.losses.Reduction.SUM_OVER_BATCH_SIZE
        )

        self.mse = tf.keras.losses.MeanSquaredError(
            reduction = tf.keras.losses.Reduction.SUM_OVER_BATCH_SIZE
        )

    def get_mask(self, y_true):
        try:
            anchor_one_mask = tf.where(
                y_true[:,:,:,:, 0] == 0,
                0.5, 
                5.0
            )

            anchor_two_mask = tf.where(
                y_true[:,:,:,:, 5] == 0,
                0.5, 
                5.0
            )

            bboxes_mask = tf.concat(
                [anchor_one_mask, anchor_two_mask],
                axis=0
            )

            return bboxes_mask
        
        except Exception as e:
            raise CustomException(e, sys)

    def call(self, y_true, y_pred):
        try:
            prob_loss = self.binary_crossentropy(
                tf.concat([y_true[:, :, :, 0], y_true[:, :, :, 5]], axis=0),
                tf.concat([y_pred[:, :, :, 0], y_pred[:, :, :, 5]], axis=0)
            )

            xy_loss = self.mse(
                tf.concat([y_true[:, :, :, 1:3], y_true[:, :, :, 6:8]], axis=0),
                tf.concat([y_pred[:, :, :, 1:3], y_pred[:, :, :, 6:8]], axis=0)
            )

            wh_loss = self.mse(
                tf.concat([y_true[:, :, :, 3:5], y_true[:, :, :, 8:10]], axis=0),
                tf.concat([y_pred[:, :, :, 3:5], y_pred[:, :, :, 8:10]], axis=0)
            )

            bboxes_mask = self.get_mask(y_true)

            xy_loss = xy_loss * bboxes_mask
            wh_loss = wh_loss * bboxes_mask

            return prob_loss + xy_loss + wh_loss
        
        except Exception as e:
            raise CustomException(e, sys)
            

    