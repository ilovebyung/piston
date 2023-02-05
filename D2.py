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

D2 = image[160:210]
plt.imshow(D2, cmap='gray')
# cv2.imshow('D', D2)
# cv2.waitKey(0)
# cv2.destroyWindow('D')

'''
overwrite images
'''

for file in files:
    if file.endswith('jpg'):
        # crop image and get D
        image = cv2.imread(file, 0)
        D1 = image[160:210]
        # set filename with _D
        filename = 'd2/' + file[:-4] + '_D2' + '.jpg'
        # plt.imshow(D1,cmap='gray')
        cv2.imwrite(filename, D1)
        print(filename)
