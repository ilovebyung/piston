import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage.metrics import structural_similarity


# Load images
before = cv2.imread('BMW - S58_20220727074428_220706075959030240234182_1_NIO.png',0)
# after = cv2.imread('bad_01.png',0)
after = cv2.imread('BMW - S58_20211125043007_211108074937030240234182_1_IO.png',0)

# # Convert images to grayscale
# before = cv2.cvtColor(before, cv2.COLOR_BGR2GRAY)
# after = cv2.cvtColor(after, cv2.COLOR_BGR2GRAY)

# Compute SSIM between the two images
(score, diff) = structural_similarity(before, after, full=True)
print("Image Similarity: {:.4f}%".format(score * 100))

# The diff image contains the actual image differences between the two images
# and is represented as a floating point data type in the range [0,1]
# so we must convert the array to 8-bit unsigned integers in the range
# [0,255] before we can use it with OpenCV
diff = (diff * 255).astype("uint8")
diff_box = cv2.merge([diff, diff, diff])

# Threshold the difference image, followed by finding contours to
# obtain the regions of the two input images that differ
thresh = cv2.threshold(
    diff, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
contours = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
contours = contours[0] if len(contours) == 2 else contours[1]

mask = np.zeros(before.shape, dtype='uint8')
filled_after = after.copy()

for object in contours:
    area = cv2.contourArea(object)
    if area > 600:  
        x, y, w, h = cv2.boundingRect(object)
        cv2.rectangle(before, (x, y), (x + w, y + h), (36, 255, 12), 2)
        cv2.rectangle(after, (x, y), (x + w, y + h), (36, 255, 12), 2)
        cv2.rectangle(diff_box, (x, y), (x + w, y + h), (36, 255, 12), 2)
        cv2.drawContours(mask, [object], 0, (255, 255, 255), -1)
        cv2.drawContours(filled_after, [object], 0, (0, 255, 0), -1)

# cv2.imshow('before', before)
# cv2.imshow('after', after)
# cv2.imshow('diff', diff)
# cv2.imshow('diff_box', diff_box)
# cv2.imshow('mask', mask)
# cv2.imshow('filled after', filled_after)
# cv2.waitKey()

plt.imshow(diff)
plt.imshow(diff_box, cmap='gray')
plt.imshow(mask, cmap='gray')
plt.imshow(filled_after, cmap='gray')


cv2.imwrite('diff_box.jpg', diff_box)
cv2.imwrite('filled_after.jpg', filled_after)


# compare images
def compare(a, b):
    '''
    subtract differences between autoencoder and reconstructed image
    '''
    a = cv2.imread(a, 0)
    b = cv2.imread(b, 0)
    # autoencoder - reconstructed
    inv_01 = cv2.subtract(a, b)

    # reconstructed - autoencoder
    inv_02 = cv2.subtract(b, a)

    # combine differences
    combined = cv2.addWeighted(inv_01, 0.5, inv_02, 0.5, 0)
    return combined

# diff = compare('good.png','bad_01.png') 
diff = compare('good.png','bad_02.png')
cv2.imshow('diff', diff)
cv2.imwrite('compared.jpg', diff)
cv2.waitKey()