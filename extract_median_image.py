import numpy as np
import cv2
import os
import matplotlib.pyplot as plt

os.chdir('/app/data/piston/d3')
files = os.listdir()
images = []

for file in files:
    if file.endswith(".jpg"):
        image = cv2.imread(file, 0)
        images.append(image)

frames = np.array(images)

# calculate the average
median_image = np.median(frames, axis=0).astype(dtype=np.uint8)
plt.imshow(median_image, cmap='gray')
cv2.imwrite("median_image.jpg", median_image)
