import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
import numpy as np
import cv2
from glob import glob
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau, CSVLogger, TensorBoard
from tensorflow.keras.metrics import Recall, Precision, IoU
from data import load_data, tf_dataset
from model import build_model
from models import create_segmentation_model
import matplotlib.pyplot as plt

def iou(y_true, y_pred):
    def f(y_true, y_pred):
        intersection = (y_true * y_pred).sum()
        union = y_true.sum() + y_pred.sum() - intersection

        x = (intersection + 1e-15) / (union + 1e-15)
        x = x.astype(np.float32)

        return x
    return tf.numpy_function(f, [y_true , y_pred], tf.float32)

if __name__ == '__main__':
    path = 'train_videos'
    (train_x, train_y), (valid_x, valid_y), (test_x, test_y) = load_data(path=path)
    print(len(train_x), len(valid_x), len(test_x))

    batch = 16
    lr = 1e-4
    epochs=50

    train_dataset = tf_dataset(train_x, train_y, batch=batch)
    valid_dataset = tf_dataset(valid_x, valid_y, batch=batch)

    model = create_segmentation_model(input_size=256, l=4)
    optimizer = tf.keras.optimizers.Adam(lr)
    metrics = ['acc', Recall, Precision, IoU]
    model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['acc', Recall(), Precision(), iou])

    callbacks = [
        ModelCheckpoint(r'files\\unetpp.h5'),
        ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=3),
        CSVLogger(r'files\\data.csv'),
        TensorBoard(),
        EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    ]

    train_steps = len(train_x) // batch
    valid_steps = len(valid_x) // batch

    if len(train_x) % batch != 0:
        train_steps += 1
    if len(valid_x) % batch != 0:
        valid_steps += 1
    history = model.fit(
        train_dataset,
        validation_data=valid_dataset,
        epochs=epochs,
        steps_per_epoch=train_steps,
        validation_steps=valid_steps,
        callbacks=callbacks
    )   