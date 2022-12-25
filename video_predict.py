import cv2
import tensorflow as tf
import numpy as np
import os
import glob
from train import iou
from tensorflow.keras.utils import CustomObjectScope

def capture_video(file_path):
    frames = []
    cap = cv2.VideoCapture(file_path)
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("Done recording")
            break
        frame = cv2.resize(frame, (256, 256))
        frame = frame / 255.0
        frames.append(frame)
    cap.release()
    frames = np.array(frames)
    return True, frames

def mask_parse(mask):
    mask = np.squeeze(mask)
    mask = [mask, mask, mask]
    mask = np.transpose(mask, (1, 2, 0))
    return mask

if __name__ == '__main__':
    video_path = r'Inference videos\ShortVD_wp_68\ShortVD_wp_68.wmv'
    _, frames = capture_video(video_path)
    print(frames.shape)

    with CustomObjectScope({'iou':iou}):
        model = tf.keras.models.load_model(r'files\model_big.h5')
    i = 0

    frame_count = [3, 13, 77]
    for frame in frames:
        '''if i == 3 or i == 13 or i == 77:
            cv2.imwrite(r'weird_frames\\' + str(i) + '.png', frame * 255.0)'''
        mask_pred = model.predict(np.expand_dims(frame, axis=0), verbose=0)
        mask_pred = mask_pred[0] > 0.5
        mask_pred = mask_parse(mask_pred) * 255.0
        cv2.imwrite(r'video_results\\' + str(i) + '.png', mask_pred)
        i = i + 1