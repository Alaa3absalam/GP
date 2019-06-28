import cv2
import glob
import h5py
import cv2 as cv
import numpy as np

import time

start = time.time()

videoList = glob.glob(r'C:\Users\Nour\.PyCharmCE2019.1\config\scratches\Egypt\*.mp4')
frames = []
caps = []

for path in videoList:
    print(path)
    caps.append(cv2.VideoCapture(path))
    for cap in caps:
        while cap.isOpened():
            ret, img = cap.read()
            #img_gray = cv.cvtColor(img, cv.COLOR_RGB2GRAY)

            if not ret:
                break
            frames.append(img)
            with h5py.File('./trainclk.hdf5', 'w') as h5File:
                h5File.create_dataset('الساعة كام', data=frames, compression='gzip', compression_opts=9)

with h5py.File('trainclk.hdf5', 'r') as dataset:
    x_train = dataset["الساعة كام"][:]
    print("الساعة كام shape: ", x_train.shape)

end = time.time()
print(end - start)