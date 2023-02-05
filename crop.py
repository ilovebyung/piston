import glob
import cv2
import matplotlib.pyplot as plt
import os

file = '/home/iot/Pictures/training/WIN_20220801_15_22_26_Pro.jpg'
a = cv2.imread(file,0)
a.shape
plt.imshow(a, cmap='gray')

crop_img = a[200:900, 50:1850]
plt.imshow(crop_img, cmap='gray')


file = '/home/iot/Pictures/board_anomaly/WIN_20220718_11_13_45_Pro.jpg'
b = cv2.imread(file,0)
b.shape
plt.imshow(b, cmap='gray')

diff = diff(a,b)
plt.imshow(diff, cmap='gray')

# filename = '20220701_201514.jpg'
# file = os.path.join(path, filename)
# img = cv2.imread(file,0)
# plt.imshow(img)

crop_img = img[1000:, 1000:]
plt.imshow(crop_img)

# cv2.imshow("cropped", crop_img)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

path = '/home/iot/Pictures/training/'
path = '/home/iot/Pictures/anomaly/'
path = '/home/iot/Pictures/board_anomaly'
cwd = '/home/iot/Pictures/training'
cwd = '/home/iot/Pictures/anomaly/'
cwd = '/home/iot/Pictures/board_anomaly'
os.chdir(cwd)

'''
convert color to gray with crop size
'''

for filename in glob.glob(os.path.join(path, '*')):
    img = cv2.imread(filename, 0)
    crop_img = img[200:900, 50:1850]
    print(filename, crop_img.shape)
    cv2.imwrite(filename, crop_img)

# prepare image A
size = (512, 512)
filename = '20220701_201514.jpg'
data = cv2.imread(filename, 0)
resized_img = cv2.resize(data, size)
resized_img.shape
cv2.imshow('resized_data', resized_img)
cv2.waitKey(0)
cv2.destroyAllWindows()

# prepare image B
decoded_file = 'decoded_img.jpg'
decoded_img = cv2.imread(decoded_file, 0)
decoded_img.shape
cv2.imshow('decoded_img', decoded_img)
cv2.waitKey(0)
cv2.destroyAllWindows()


def diff(a, b):
    '''
    subtract differences between autoencoder and reconstructed image
    '''
    # set the same size
    # a = cv2.resize(a, (512, 512))
    # b = cv2.resize(b, (512, 512))

    # autoencoder - reconstructed
    inv_01 = cv2.subtract(a, b, dtype = cv2.CV_32F)

    # reconstructed - autoencoder
    inv_02 = cv2.subtract(b, a, dtype = cv2.CV_32F)

    # combine differences
    combined = cv2.addWeighted(inv_01, 0.5, inv_02, 0.5, 0)
    return combined



A = os.path.join('C:\source\card\diff', 'a.jpg')
A = cv2.imread(A, 0)

B = os.path.join('C:\source\card\diff', 'B.jpg')
B = cv2.imread(B, 0)

plt.imshow(diff(A, B))
diff_img = diff(A, B)
cv2.imwrite('diff_img.jpg', diff_img)

import numpy as np
a = np.array(9)