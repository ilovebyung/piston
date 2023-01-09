'''
1. Create training data
2. Build autoencoder 
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

'''
Set size of images
'''
path = '/home/m0034463/store/data/piston/d0'
# path = '/home/m0034463/store/data/piston/equipped_io/d1'
# path = '/home/m0034463/store/data/piston/equipped_io/d2'
# path = '/home/m0034463/store/data/piston/equipped_io/d3'

os.chdir(path)
files = os.listdir()
# size = (631, 3202)
image = cv2.imread(files[0],0)
size = image.shape
height = size[0]
width = size[1]


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
latent_dim = 64 * 3

class Autoencoder(Model):
  def __init__(self, latent_dim):
    super(Autoencoder, self).__init__()
    self.latent_dim = latent_dim   
    self.encoder = tf.keras.Sequential([
        Flatten(),
        Dense(latent_dim, activation='relu'),
        Dense(latent_dim, activation='relu'),
        Dense(latent_dim, activation='relu'),
    ])
    self.decoder = tf.keras.Sequential([
        Dense(latent_dim, activation='relu'),      
        Dense(latent_dim, activation='relu'),     
        Dense(height * width, activation='sigmoid'),
        Reshape((height, width))
    ])

  def call(self, x):
    encoded = self.encoder(x)
    decoded = self.decoder(encoded)
    return decoded


if __name__ == "__main__":

    '''
    1. Create training data  
    '''

    # path = '/home/m0034463/store/data/piston/equipped_io/d0'
    # path = '/home/m0034463/store/data/piston/equipped_io/d1'
    # path = '/home/m0034463/store/data/piston/equipped_io/d2'
    path = '/home/m0034463/store/data/piston/equipped_io/d3'

    data = create_training_data(path, height=height, width=width)
    x_train = data[:-10]
    x_test = data[-10:]

    x_train = x_train.reshape(-1,height, width,1)
    x_test = x_test.reshape(-1,height, width,1)

    # check the integrity of an image
    img = x_test[0]
    plt.imshow(img, cmap='gray')

    '''
    2. Build autoencoder  
    '''
    # autoencoder = Autoencoder()
    autoencoder = Autoencoder(latent_dim)
    autoencoder.compile(optimizer='adam', loss=losses.MeanSquaredError())

    history = autoencoder.fit(x_train, x_train,
                            epochs=60,
                            shuffle=True,
                            validation_data=(x_test, x_test))

    # plot history
    plt.plot(history.history["loss"], label="Training Loss")
    plt.plot(history.history["val_loss"], label="Validation Loss")
    plt.legend()
    plt.show()



    # check the integrity of predictions
    x_decoded = autoencoder.predict(x_test) 
    img = x_decoded[0]
    plt.imshow(img, cmap='gray')

    # check the location of model to be saved
    os.getcwd()

    # save a mode
    autoencoder.save('./model/')

    # load autoencoder model
    try:
        os.chdir(path)
        dir(autoencoder) 
    except:
        autoencoder = keras.models.load_model('./model/')

