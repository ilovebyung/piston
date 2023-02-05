import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage.metrics import structural_similarity


# Load images
before = cv2.imread('01.jpg', 0)
width, height = before.shape
dim = (height, width)
after = cv2.imread('13.jpg', 0)

# # Convert images to grayscale
# before_gray = cv2.cvtColor(before, cv2.COLOR_BGR2GRAY)
# after_gray = cv2.cvtColor(after, cv2.COLOR_BGR2GRAY)

# Compute SSIM between the two images
(score, diff) = structural_similarity(before, after, full=True)
print("Image Similarity: {:.4f}%".format(score * 100))

# compare images
def diff(a, b):
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
    plt.imshow(combined, cmap='gray')
    # return combined


a = '01.jpg'
b = '03.jpg'

diff(a, b)

for file in files:
    image = cv2.imread(file, 0)
    (score, diff) = structural_similarity(before, image, full=True)
    print(file, "Image Similarity: {:.4f}%".format(score * 100))


# rename filename and extension

path = 'C:/data/piston/train/equipped'
path = 'C:/data/piston/train/unequipped'
path = 'C:/data/piston/test/equipped'
path = 'C:/data/piston/test/unequipped'
path = 'C:/data/piston/bad/equipped not ok'
path = 'C:/data/piston/bad/unequipped not ok'

os.chdir(path)
files = os.listdir()

for file in files:
    if file.endswith('png'):
        filename = file[:-4] + '.jpg'
        image = cv2.imread(file, 0)
        cv2.imwrite(filename, image)
        print(filename)


