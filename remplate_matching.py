import cv2
import numpy as np
import matplotlib.pyplot as plt

# Read the background image as gray scale
bg = cv2.imread('bg.jpg', 0)
img2 = bg.copy()
plt.imshow(bg, cmap='gray')
# Read the template
template = cv2.imread('template.jpg', 0)
plt.imshow(template, cmap='gray')

# Store width and height of template in w and h
w, h = template.shape[::-1]

# Red color in BGR
color = (250, 100, 100)

# Perform match operations.
# All the 6 methods for comparison in a list
methods = ['cv2.TM_CCOEFF', 'cv2.TM_CCOEFF_NORMED', 'cv2.TM_CCORR',
           'cv2.TM_CCORR_NORMED', 'cv2.TM_SQDIFF', 'cv2.TM_SQDIFF_NORMED']

# result = cv2.matchTemplate(bg, template, cv2.TM_CCOEFF_NORMED)
result = cv2.matchTemplate(bg, template, cv2.TM_CCORR_NORMED)
plt.imshow(result, cmap='gray')

min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)

# Specify a threshold
threshold = 0.98

# Store the coordinates of matched area in a numpy array
loc = np.where(result >= threshold)

# Draw a rectangle around the matched region.
for pt in zip(*loc[::-1]):
    cv2.rectangle(bg, pt, (pt[0] + w, pt[1] + h), color, 2)

# Show the final image with the matched area.
cv2.imshow('M shape', bg)
cv2.waitKey()
cv2.destroyWindows()


# Window name in which image is displayed
window_name = 'M shape'

# text
text = 'x'

# font
font = cv2.FONT_HERSHEY_SIMPLEX

# fontScale
fontScale = 1

# Line thickness of 2 px
thickness = 2

# Using cv2.putText() method
image = cv2.putText(bg, text, max_loc, font, fontScale,
                    color, thickness, cv2.LINE_AA, False)

# # Using cv2.putText() method
# image = cv2.putText(image, text, max_loc, font, fontScale,
#                     color, thickness, cv2.LINE_AA, True)

# Displaying the image
cv2.imshow(window_name, image)
cv2.waitKey()
cv2.destroyWindows()
