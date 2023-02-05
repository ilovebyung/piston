import os
import cv2
import numpy as np
import matplotlib.pyplot as plt


'''
resize images
'''
path = 'C:/data/piston/bad/equipped_io/d0'
path = 'C:/data/piston/bad/equipped_io/d1'
path = 'C:/data/piston/bad/equipped_io/d2'
path = 'C:/data/piston/bad/equipped_io/d3'
path = 'C:/data/piston/bad/equipped_nio/d0'
path = 'C:/data/piston/bad/equipped_nio/d1'
path = 'C:/data/piston/bad/equipped_nio/d2'
path = 'C:/data/piston/bad/equipped_nio/d3'
os.chdir(path)
files = os.listdir()

# get image dimension
filename = files[0]
image = cv2.imread(filename, 0)
height, width = image.shape
plt.imshow(image, cmap='gray')

# trim images
trimmed = image[20:, :]
plt.imshow(trimmed, cmap='gray')

for file in files:
    if file.endswith('jpg'):
        image = cv2.imread(file, 0)
        resized = image[20:, :]
        cv2.imwrite(file, resized)
        print(file)


# resize images

height, width = (632, 3208)  # d0
height, width = (72, 3208)  # d1
height, width = (56, 3208)  # d2
height, width = (56, 3208)  # d3

dim = int(width/2), int(height/2)  # mind the order
for file in files:
    if file.endswith('jpg'):
        image = cv2.imread(file, 0)
        resized = cv2.resize(image, dim, interpolation=cv2.INTER_AREA)
        cv2.imwrite(file, resized)
        print(file)
