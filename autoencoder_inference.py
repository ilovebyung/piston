'''
3. Make an inference
4. Find differences
'''
from tensorflow.keras.models import Model
from sklearn.metrics import accuracy_score, precision_score, recall_score
from tensorflow.keras.layers import Dense, Input
from tensorflow.python.keras import losses
from tensorflow.keras.layers import Conv2D, Flatten, Reshape, Conv2DTranspose
from tensorflow.keras.utils import plot_model
from tensorflow.keras.layers import MaxPooling2D, UpSampling2D
import tensorflow.keras as keras
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import cv2
import os
import pathlib

gpu_available = tf.config.list_physical_devices('GPU')
print(tf.test.is_gpu_available(cuda_only=True))

def get_image_loss(file):
    '''
    Load an image from a file and calculate sample loss
    '''
    # data = np.ndarray(shape=(1, size, size), dtype=np.float32)

    data = cv2.imread(str(file), 0)
    # resize to make sure data consistency
    resized_data = cv2.resize(data, (width, height))
    # nomalize img
    normalized_data = resized_data.astype('float32') / 255.
    # test an image
    encoded = autoencoder.encoder(normalized_data.reshape(-1, height, width))
    decoded = autoencoder.decoder(encoded)
    loss = tf.keras.losses.mse(decoded, normalized_data)
    sample_loss = np.mean(loss) + 3*np.std(loss)
    return sample_loss

def get_decoded_img(file):
    '''
    Generate a decoded image and return it as grayscale integer
    '''
    image = cv2.imread(file, 0)
    resized = cv2.resize(image, (width, width), interpolation=cv2.INTER_AREA)
    reshaped = resized.reshape(-1, height, width)
    encoded_img = autoencoder.encoder(reshaped).numpy()
    decoded_img = autoencoder.decoder(encoded_img).numpy()
    reconstructed_img = decoded_img.reshape(height, width)*255
    # plt.imshow(decoded_img.reshape(224,224), cmap='gray')
    return reconstructed_img.astype(int)

# compare images
def show_difference(a, b):
    '''
    subtract differences between autoencoder and reconstructed image
    '''
    if (type(a) is str) and (type(b) is str):
        a = cv2.imread(a, 0)
        b = cv2.imread(b, 0)
        # autoencoder - reconstructed
        inv_01 = cv2.subtract(a, b)

        # reconstructed - autoencoder
        inv_02 = cv2.subtract(b, a)

        # combine differences
        combined = cv2.addWeighted(inv_01, 0.5, inv_02, 0.5, 0)
        return combined

    if (type(a) is np.ndarray) and (type(b) is np.ndarray):
        a = a.astype(int)
        b = b.astype(int)
        # autoencoder - reconstructed
        inv_01 = cv2.subtract(a, b)

        # reconstructed - autoencoder
        inv_02 = cv2.subtract(b, a)

        # combine differences
        combined = cv2.addWeighted(inv_01, 0.5, inv_02, 0.5, 0)
        return combined

if __name__ == "__main__":

    '''
    Load corresponding model
    '''

    # path = '/home/m0034463/store/data/piston/d0'
    # path = '/home/m0034463/store/data/piston/d1'
    # path = '/home/m0034463/store/data/piston/d2'
    path = '/home/m0034463/store/data/piston/d3'

    os.chdir(path)
    files = os.listdir()
    image = cv2.imread(files[0],0)
    size = image.shape
    height= size[0]
    width = size[1]

    # load model
    os.chdir(path)
    autoencoder = keras.models.load_model('./model/')


    '''
    3. Make an inference
    '''
    # get statistics of good parts

    values = []
    # path = '/home/m0034463/store/data/piston/d0'
    # path = '/home/m0034463/store/data/piston/d1'
    # path = '/home/m0034463/store/data/piston/d2'
    path = '/home/m0034463/store/data/piston/d3'

    os.chdir(path)
    files = os.listdir()
    for filename in files:
        # check file extention
        if filename.endswith(".jpg"):
            data = cv2.imread(filename, 0)
            # resize to make sure data consistency
            resized_data = cv2.resize(data, (width, height), interpolation=cv2.INTER_AREA) 
            # nomalize img
            normalized_data = resized_data.astype('float32') / 255.
            # test an image
            encoded = autoencoder.encoder(normalized_data.reshape(-1, height, width,1))
            decoded = autoencoder.decoder(encoded)
            loss = tf.keras.losses.mse(decoded, normalized_data.reshape(-1, height, width))
            sample_loss = np.mean(loss) + np.std(loss)
            values.append(sample_loss)
            print("IO parts sample loss: ", sample_loss)  

    plt.xlabel('loss value')
    plt.ylabel('number of samples')
    plt.hist(values)  

    print(f'mean: {np.mean(values)}, median: {np.median(values)}')


    # bad parts
    values = []
    # path = '/home/m0034463/store/data/piston/equipped_nio/d0'
    # path = '/home/m0034463/store/data/piston/equipped_nio/d1'
    # path = '/home/m0034463/store/data/piston/equipped_nio/d2'
    path = '/home/m0034463/store/data/piston/equipped_nio/d3'

    os.chdir(path)
    files = os.listdir()
    for filename in files:
        # check file extention
        if filename.endswith(".jpg"):
            data = cv2.imread(filename, 0)
            # resize to make sure data consistency
            resized_data = cv2.resize(data, (width, height), interpolation=cv2.INTER_AREA) 
            # nomalize img
            normalized_data = resized_data.astype('float32') / 255.
            # test an image
            encoded = autoencoder.encoder(normalized_data.reshape(-1, height, width,1))
            decoded = autoencoder.decoder(encoded)
            loss = tf.keras.losses.mse(decoded, normalized_data.reshape(-1, height, width))
            sample_loss = np.mean(loss) + np.std(loss)
            values.append(sample_loss)
            print("NIO parts sample loss: ", sample_loss)  

    plt.xlabel('loss value')
    plt.ylabel('number of samples')
    plt.hist(values)    

    threshold = 0.05
    anomaly = []

    for file, value in zip (files, values):
        if value > threshold:
            print (file, value)
            anomaly.append(file)

    import csv

    # check images
    filename = anomaly[0]
    data = cv2.imread(filename, 0)
    resized_data = cv2.resize(data, (width, height), interpolation=cv2.INTER_AREA) 
    normalized_data = resized_data.astype('float32') / 255.
    encoded = autoencoder.encoder(normalized_data.reshape(-1, height, width))
    decoded = autoencoder.decoder(encoded)
    img = decoded.numpy()
    plt.imshow(img.reshape(height,width), cmap='gray')

    '''
    4. Find differences
    '''
    # bad parts  
    filename = anomaly[0]

    image = cv2.imread(filename, 0)
    resized_data = cv2.resize(image, (width, height))
    resized_data.shape
    expanded_data = tf.expand_dims(resized_data, axis=0)
    expanded_data.shape
    encoded_img = autoencoder.encoder(expanded_data).numpy()
    encoded_img.shape
    decoded_img = autoencoder.decoder(encoded_img).numpy()
    decoded_img.shape
    reconstructed = np.squeeze(decoded_img, 0)
    reconstructed.shape
    plt.imshow(reconstructed, cmap='gray')

    # Load two images
    resized_data.shape
    reconstructed *= 255
    reconstructed.shape
    reconstructed = reconstructed.astype(int)
    cv2.imwrite('/home/m0034463/store/data/piston/reconstructed.jpg', reconstructed)



    diff = show_difference(image, reconstructed)
    # cv2.imshow('diff', diff)
    cv2.imwrite('/home/m0034463/store/data/piston/compared.jpg', diff)



    reconstructed = cv2.imread('/home/m0034463/store/data/piston/reconstructed.jpg')
    plt.imshow(reconstructed)


