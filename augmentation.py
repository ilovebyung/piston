from keras.preprocessing.image import ImageDataGenerator
import numpy as np
import cv2

# Define directory
source = 'D:/data/sonic_source/357.jpeg'
target = 'D:/data/sonic_augmentation'

# Define generator
datagen = ImageDataGenerator(
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest')

img = cv2.imread(source)
img = img.reshape((1,) + img.shape)
i = 0
for batch in datagen.flow(img, batch_size=1,
                          save_to_dir=target, save_prefix='image', save_format='jpeg'):
    i += 1
    if i > 20:
        break
