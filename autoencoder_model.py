from tensorflow.keras.models import Model
from sklearn.metrics import accuracy_score, precision_score, recall_score
from tensorflow.keras.layers import Dense, Input
from tensorflow.python.keras import losses
from tensorflow.keras.layers import Conv2D, Flatten, Reshape, Conv2DTranspose
from tensorflow.keras.utils import plot_model
from tensorflow.keras.layers import MaxPooling2D, UpSampling2D
from tensorflow.keras import layers, losses
import tensorflow.keras as keras
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import cv2
import os 


gpu_available = tf.config.list_physical_devices('GPU')
print(tf.test.is_gpu_available(cuda_only=True))

'''
Set size of images
'''
path = '/home/m0034463/store/data/piston/equipped_io/d0'
# path = '/home/m0034463/store/data/piston/equipped_io/d1'
# path = '/home/m0034463/store/data/piston/equipped_io/d2'
# path = '/home/m0034463/store/data/piston/equipped_io/d3'

os.chdir(path)
files = os.listdir()
image = cv2.imread(files[0],0)
size = image.shape
width= size[0]
height = size[1]

'''
Autoencoder Models
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

    
class Autoencoder(Model):
  def __init__(self):
    super(Autoencoder, self).__init__()
    self.encoder = tf.keras.Sequential([
      layers.Input(shape=(height, width, 1)),
      layers.Conv2D(16*2, (3, 3), activation='relu', padding='same', strides=2),
      layers.Conv2D(16, (3, 3), activation='relu', padding='same', strides=2),
      layers.Conv2D(8, (3, 3), activation='relu', padding='same', strides=2)])

    self.decoder = tf.keras.Sequential([
      layers.Conv2DTranspose(8, kernel_size=3, strides=2, activation='relu', padding='same'),
      layers.Conv2DTranspose(16, kernel_size=3, strides=2, activation='relu', padding='same'),
      layers.Conv2DTranspose(16*2, kernel_size=3, strides=2, activation='relu', padding='same'), 
      layers.Conv2D(1, kernel_size=(3, 3), activation='sigmoid', padding='same')])

  def call(self, x):
    encoded = self.encoder(x)
    decoded = self.decoder(encoded)
    return decoded
