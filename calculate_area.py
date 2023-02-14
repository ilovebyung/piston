import glob
import cv2
import matplotlib.pyplot as plt
import os

# Load image
file = './test/equipped/BMW - S58_20211125043007_211108074937030240234182_1_IO.jpg'
image = cv2.imread(file, 0)
image.shape
plt.imshow(image, cmap='gray')

# Apply Otsu thresholding to obtain binary image
_, thresh = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

# Apply morphological operations to clean up binary image
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
cleaned = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
plt.imshow(cleaned, cmap='gray')

# Calculate area of object
area = cv2.countNonZero(cleaned)

print("Object area: ", area)
