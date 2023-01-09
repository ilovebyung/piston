'''
1. Create training data
2. Build autoencoder 
3. Make an inference
4. Find differences
'''

import pathlib
import os
import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tensorflow import keras
import tensorflow.keras as keras
from sklearn.metrics import accuracy_score, precision_score, recall_score
from tensorflow.python.keras import layers, losses
from tensorflow.python.keras.models import Model
from tensorflow.python.keras.layers.core import Dropout
import tensorflow as tf
gpu_available = tf.config.list_physical_devices('GPU')
print(tf.test.is_gpu_available(cuda_only=True))

'''
Set size of images
'''
size = (631, 3202)
height = size[0]
width = size[1]

size = 400
dim = (631, 3202)
height = dim[0]
width = dim[1]

'''
1. Create training data  
'''

def create_training_data(data_path, height, width):
    '''
    resize, normalize, and return grayscale images for training
    Mind size (height, width)
    '''

    training_data = []

    # iterate over each image
    for image in os.listdir(data_path):
        # check file extention
        if image.endswith(".jpg"):
            try:
                data_path = pathlib.Path(data_path)
                full_name = str(pathlib.Path.joinpath(data_path, image))
                data = cv2.imread(full_name, 0)
                # resize to make sure data consistency
                resized_data = cv2.resize(data, (width, height), interpolation = cv2.INTER_AREA)
                # add this to our training_data
                training_data.append([resized_data])
            except Exception as err:
                print("an error has occured: ", err, full_name)

    # normalize data
    training_data = np.array(training_data)/255.
    # reshape
    training_data = np.array(training_data).reshape(-1, height, width)
    return training_data


'''
2. Build autoencoder 
'''
# Define a convolutional Autoencoder
# size is a tuple (height,width)


class Autoencoder(Model):
    def __init__(self):
        super(Autoencoder, self).__init__()
        # input layer
        self.encoder = tf.keras.Sequential([
            layers.Flatten(),
            layers.Dense(256, activation='relu'),
            layers.Dense(128, activation='relu'),
            layers.Dense(64, activation='relu'),            
            layers.Dense(32, activation='relu'),
        ])
        self.decoder = tf.keras.Sequential([
            layers.Dense(64, activation='relu'), 
            layers.Dense(128, activation='relu'),
            layers.Dense(256, activation='relu'),
            layers.Dense(height*width, activation='sigmoid'),
            layers.Reshape((height, width))
        ])

    def call(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

# latent_dim = 64 * 4

# class Autoencoder(Model):
#     def __init__(self, latent_dim):
#         super(Autoencoder, self).__init__()
#         self.latent_dim = latent_dim   
#         self.encoder = tf.keras.Sequential([
#         layers.Flatten(),
#         layers.Dense(latent_dim, activation='relu'),
#         ])
#         self.decoder = tf.keras.Sequential([
#         layers.Dense(height*width, activation='sigmoid'),
#         layers.Reshape((height, width))
#         ])

#     def call(self, x):
#         encoded = self.encoder(x)
#         decoded = self.decoder(encoded)
#         return decoded

'''
3. Make an inference
'''
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

def generate_decoded_img(file):
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

if __name__ == "__main__":

    import time
    start = time.perf_counter()

    finish = time.perf_counter()
    print(f'Finished in {round(finish-start, 2)} second(s)')

    '''
    1. Load training images
    '''


    data_path = '/home/m0034463/store/source/piston/train/equipped/'
    x_train = create_training_data(data_path, height=height, width=width)

    data_path = '/home/m0034463/store/source/piston/test/equipped/'
    x_test = create_training_data(data_path, height=height, width=width)

    x_train = x_train.reshape(-1,height, width,1)
    x_test = x_test.reshape(-1,height, width,1)

    '''
    2. Build autoencoder 
    '''
    autoencoder = Autoencoder()
    # autoencoder = Autoencoder(latent_dim)
    autoencoder.compile(optimizer='adam', loss=losses.MeanSquaredError())

    history = autoencoder.fit(x_train, x_train,
                            epochs=40,
                            shuffle=True,
                            validation_data=(x_test, x_test))

    # plot history
    plt.plot(history.history["loss"], label="Training Loss")
    plt.plot(history.history["val_loss"], label="Validation Loss")
    plt.legend()
    plt.show()


    # predict the autoencoder output from test data
    x_decoded = autoencoder.predict(x_test)
    cv2.imshow('decoded', x_decoded[0])
    cv2.waitKey(0)

    # display the 1st 8 test input and decoded images
    imgs = np.concatenate([x_test[:8].reshape(-1,height,width,1), x_decoded[:8].reshape(-1,height,width,1)])
    imgs = imgs.reshape((4, 4, height, width))
    imgs = np.vstack([np.hstack(i) for i in imgs], cmap='gray')

    # save a mode
    autoencoder.save('./model/')

    # load autoencoder model
    try:
        dir(autoencoder) 
    except:
        autoencoder = Autoencoder()
        autoencoder = keras.models.load_model('./model/')


    '''
    4. Make an inference
    '''
    # get statistics  
    # good parts

    items = []
    path ='/home/m0034463/store/data/piston/train/equipped/'
    # path = '/home/m0034463/store/data/piston/test/equipped/'

    os.chdir(path)
    for filename in os.listdir(path):
        # check file extention
        if filename.endswith(".jpg") or filename.endswith(".png"):
            data = cv2.imread(filename, 0)
            # resize to make sure data consistency
            resized_data = cv2.resize(data, (width, height))
            # nomalize img
            normalized_data = resized_data.astype('float32') / 255.
            # test an image
            encoded = autoencoder.encoder(normalized_data.reshape(-1, height, width))
            decoded = autoencoder.decoder(encoded)
            loss = tf.keras.losses.mse(decoded, normalized_data)
            sample_loss = np.mean(loss) + np.std(loss)
            items.append(sample_loss)
            print("good parts sample loss: ", sample_loss)  

    plt.xlabel('loss value')
    plt.ylabel('number of samples')
    plt.hist(items)  

    print(f'mean: {np.mean(items)}, median: {np.median(items)}')

    # good = items

    # bad parts
    items = []
    path ='/home/m0034463/store/data/piston/bad/equipped not ok'

    os.chdir(path)
    for filename in os.listdir(path):
        # check file extention
        if filename.endswith(".jpg") or filename.endswith(".png"):
            data = cv2.imread(filename, 0)
            # resize to make sure data consistency
            resized_data = cv2.resize(data, (width, height))
            # nomalize img
            normalized_data = resized_data.astype('float32') / 255.
            # test an image
            encoded = autoencoder.encoder(normalized_data.reshape(-1, height, width))
            decoded = autoencoder.decoder(encoded)
            loss = tf.keras.losses.mse(decoded, normalized_data)
            sample_loss = np.mean(loss) + np.std(loss)
            items.append(sample_loss)
            print("bad parts sample loss: ", sample_loss)  

    plt.xlabel('loss value')
    plt.ylabel('number of samples')
    plt.hist(items)   

    print(f'mean: {np.mean(items)}, median: {np.median(items)}')


    if sample_loss > threshold:
        print(
            f'Loss is bigger than threshold \n \
              Sample Loss: {sample_loss} \n \
              Threshold: {threshold} ')
    else:
        print(
            f'Loss is smaller than threshold \n \
              Sample Loss: {sample_loss} \n \
              Threshold: {threshold} ')

    '''
    5. Find differences
    '''
    # good parts
    filename = '/app/data/piston/valid/equipped not ok/BMW - S58_20211125035657_211109064609030240234182_1_NIO.png'
    # bad parts  
    filename = '/app/data/piston/valid/equipped not ok/BMW - S58_20211125073356_211108124552030240234182_1_NIO.png'

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
    cv2.imwrite('reconstructed.jpg', reconstructed)

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

    diff = show_difference(image, reconstructed)
    # cv2.imshow('diff', diff)
    cv2.imwrite('compared.jpg', diff)
    cv2.waitKey()

    reconstructed = cv2.imread('reconstructed.jpg')
    plt.imshow(reconstructed)