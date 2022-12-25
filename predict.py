import os
import tensorflow as tf
import numpy as np
import cv2
from data import load_data, tf_dataset
from model import build_model
from tqdm import tqdm
from tensorflow.keras.utils import CustomObjectScope
from train import iou

def read_image(path):
    x = cv2.imread(path, cv2.IMREAD_COLOR)
    x = cv2.resize(x, (256, 256))
    x = x / 255.0
    return x
def read_mask(path):
    x =  cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    x = cv2.resize(x, (256, 256))
    x = np.expand_dims(x, axis=-1)
    return x

def mask_parse(mask):
    mask = np.squeeze(mask)
    mask = [mask, mask, mask]
    mask = np.transpose(mask, (1, 2, 0))
    return mask


if __name__ == '__main__':
    path = 'train_videos'
    batch = 8
    print('Predicting')
    (train_x, train_y), (valid_x, valid_y), (test_x, test_y) = load_data(path=path)
    print(len(train_x), len(valid_x), len(test_x))

    test_dataset = tf_dataset(test_x, test_y, batch=batch)
    test_steps = len(test_x) // batch
    if len(test_x) % batch != 0:
        test_steps += 1
    
    with CustomObjectScope({'iou':iou}):
        model = tf.keras.models.load_model(r'files\model_big.h5')
    
    print(model.evaluate(test_dataset, steps=test_steps))

    for i, (x, y) in tqdm(enumerate(zip(test_x, test_y)), total=len(test_x)):
        x = read_image(x)
        y = read_mask(y)
        y_pred = model.predict(np.expand_dims(x, axis=0))
        y_pred = y_pred[0] > 0.5
        h, w, _ = x.shape

        white_line = np.ones((h, 10, 3)) * 255.0

        all_images = [
            x * 255.0, white_line,
            mask_parse(y), white_line,
            mask_parse(y_pred) * 255.0
        ]
        image = np.concatenate(all_images,axis=1)
        cv2.imwrite(r'results\\' + str(i) + '.png', image)