from keras.preprocessing.image import ImageDataGenerator
import numpy as np
import cv2
import os
import glob
import matplotlib.pyplot as plt

# Define generator
datagen = ImageDataGenerator(
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest')

# Define working directory
source = 'D:/data/sonic_source'
target = 'D:/data/sonic_augmentation'

os.chdir(source)
files = glob.glob('*.jpeg')

for file in files:
    # img = cv2.imread(source)
    img = plt.imread(file)
    img = img.reshape((1,) + img.shape)
    i = 0
    for batch in datagen.flow(img, batch_size=1,
                              save_to_dir=target, save_prefix='image', save_format='jpeg'):
        i += 1
        if i > 20:
            break
