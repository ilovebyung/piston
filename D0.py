import cv2
import os
import numpy as np
import matplotlib.pyplot as plt

path = 'C:/data/piston/equipped_io'
path = 'C:/data/piston/equipped_nio'

os.chdir(path)
files = os.listdir()

image = cv2.imread(files[0], 0)
plt.imshow(image, cmap='gray')
image.shape
# cv2.imshow('original', image)
# cv2.waitKey(0)

# D1 = image[60:130]
# plt.imshow(D1, cmap='gray')
# cv2.imshow('D', D1)
# cv2.waitKey(0)
# cv2.destroyWindow('D')

# D2 = image[160:210]
# # plt.imshow(D2, cmap='gray')
# cv2.imshow('D', D2)
# cv2.waitKey(0)
# cv2.destroyWindow('D')

# D3 = image[220:270]
# # plt.imshow(D3, cmap='gray')
# cv2.imshow('D', D3)
# cv2.waitKey(0)
# cv2.destroyWindow('D')

'''
overwrite images
'''

for file in files:
    if file.endswith('jpg'):
        image = cv2.imread(file, 0)
        mask = np.zeros(image.shape, dtype="uint8")
        cv2.rectangle(mask, (0, 0), (3202, 60), 255, -1)
        cv2.rectangle(mask, (0, 130), (3202, 160), 255, -1)
        cv2.rectangle(mask, (0, 210), (3202, 220), 255, -1)
        cv2.rectangle(mask, (0, 270), (3202, 620), 255, -1)
        # apply mask -- notice how only section C is cropped out
        masked = cv2.bitwise_and(image, image, mask=mask)
        filename = 'd0/' + file[:-4] + '_D0' + '.jpg'
        cv2.imwrite(filename, masked)
        print(filename)


