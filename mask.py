import cv2
import os
import numpy as np
import matplotlib.pyplot as plt

path = 'C:/data/piston/bad/equipped not ok'

os.chdir(path)
files = os.listdir()

image = cv2.imread(files[0], 0)
plt.imshow(image, cmap='gray')
image.shape
# cv2.imshow('original', image)
# cv2.waitKey(0)


# a mask is the same size as our image, but has only two pixel
# values, 0 and 255 -- pixels with a value of 0 (background) are
# ignored in the original image while mask pixels with a value of
# 255 (foreground) are allowed to be kept
mask = np.zeros(image.shape, dtype="uint8")
plt.imshow(mask, cmap='gray')
cv2.rectangle(mask, (0, 0), (3202, 300), 255, -1)
cv2.rectangle(mask, (100, 310), (740, 631), 255, -1)
cv2.rectangle(mask, (1700, 310), (2300, 631), 255, -1)
plt.imshow(mask, cmap='gray')
# apply mask -- notice how only section C is cropped out
masked = cv2.bitwise_and(image, image, mask=mask)
plt.imshow(masked, cmap='gray')

# cv2.imshow('masked', masked)
# cv2.waitKey(0)

'''
overwrite images
'''

path = 'C:/data/piston/train/equipped'
# path = 'C:/data/piston/train/unequipped'
path = 'C:/data/piston/test/equipped'
# path = 'C:/data/piston/test/unequipped'
path = 'C:/data/piston/bad/equipped not ok'
# path = 'C:/data/piston/bad/unequipped not ok'


os.chdir(path)
files = os.listdir()

for file in files:
    if file.endswith('jpg'):
        image = cv2.imread(file, 0)
        masked = cv2.bitwise_and(image, image, mask=mask)
        cv2.imwrite(file, masked)
        print(file)
