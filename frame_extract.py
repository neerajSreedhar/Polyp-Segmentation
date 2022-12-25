from sys import path
import cv2
import os
import glob
import numpy as np
from pathlib import Path
def capture_frames(file_path):
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

if __name__ == '__main__':
    video_path = r'train_videos\videos'
    frame_sav_dir = r'train_videos\frames'
    video_paths = glob.glob(os.path.join(video_path, '*.wmv'))
    for paths in video_paths:
        file_name = Path(paths).stem
        _, frames = capture_frames(paths)
        for i in range(len(frames)):
            save_path = frame_sav_dir + '\\' + file_name + '_frame_' + str(i + 1) + '_GT.png'
            cv2.imwrite(save_path, frames[i] * 255.0)
        print('Saved frames for video: ', file_name)