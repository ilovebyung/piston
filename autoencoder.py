'''
1. Create training data
2. Build autoencoder 
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

'''
Set size of images
'''
# path = '/home/m0034463/store/data/piston/equipped_io/d0'
# path = '/home/m0034463/store/data/piston/equipped_io/d1'
# path = '/home/m0034463/store/data/piston/equipped_io/d2'
path = '/home/m0034463/store/data/piston/equipped_io/d3'

os.chdir(path)
files = os.listdir()
# size = (631, 3202)
image = cv2.imread(files[0],0)
size = image.shape
width= size[0]
height = size[1]


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

    
# from tensorflow.keras import layers, losses
# class Autoencoder(Model):
#   def __init__(self):
#     super(Autoencoder, self).__init__()
#     self.encoder = tf.keras.Sequential([
#       layers.Input(shape=(height, width, 1)),
#       layers.Conv2D(16*2, (3, 3), activation='relu', padding='same', strides=2),
#       layers.Conv2D(16, (3, 3), activation='relu', padding='same', strides=2),
#       layers.Conv2D(8, (3, 3), activation='relu', padding='same', strides=2)])

#     self.decoder = tf.keras.Sequential([
#       layers.Conv2DTranspose(8, kernel_size=3, strides=2, activation='relu', padding='same'),
#       layers.Conv2DTranspose(16, kernel_size=3, strides=2, activation='relu', padding='same'),
#       layers.Conv2DTranspose(16*2, kernel_size=3, strides=2, activation='relu', padding='same'), 
#       layers.Conv2D(1, kernel_size=(3, 3), activation='sigmoid', padding='same')])

#   def call(self, x):
#     encoded = self.encoder(x)
#     decoded = self.decoder(encoded)
#     return decoded



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

    '''
    1. Load training images
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



    # predict the autoencoder output from test data
    x_decoded = autoencoder.predict(x_test) 
    
    img = x_decoded[0]
    plt.imshow(img, cmap='gray')

    # save a mode
    autoencoder.save('./model/')

    # load autoencoder model
    try:
        os.chdir(path)
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
    path ='/home/m0034463/store/source/piston/train/equipped'

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
            loss = tf.keras.losses.mse(decoded, normalized_data.reshape(-1, height, width,1))
            sample_loss = np.mean(loss) + np.std(loss)
            items.append(sample_loss)
            print("IO parts sample loss: ", sample_loss)  

    plt.xlabel('loss value')
    plt.ylabel('number of samples')
    plt.hist(items)  

    print(f'mean: {np.mean(items)}, median: {np.median(items)}')


    # bad parts
    items = []
    path ='/home/m0034463/store/source/piston/bad/equipped not ok'

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
            loss = tf.keras.losses.mse(decoded, normalized_data.reshape(-1, height, width,1))
            sample_loss = np.mean(loss) + np.std(loss)
            items.append(sample_loss)
            print("NIO parts sample loss: ", sample_loss)  

    plt.xlabel('loss value')
    plt.ylabel('number of samples')
    plt.hist(items)    

    print(f'mean: {np.mean(items)}, median: {np.median(items)}')

    # check images
    filename = files[10]
    data = cv2.imread(filename, 0)
    resized_data = cv2.resize(data, (width, height), interpolation=cv2.INTER_AREA) 
    normalized_data = resized_data.astype('float32') / 255.
    encoded = autoencoder.encoder(normalized_data.reshape(-1, height, width,1))
    decoded = autoencoder.decoder(encoded)
    img = decoded.numpy()
    cv2.imshow('decoded', img.reshape(width,height))
    cv2.waitKey(0)

    '''
    5. Find differences
    '''
    # good parts
    filename = '/app/data/store/valid/equipped not ok/BMW - S58_20211125035657_211109064609030240234182_1_NIO.png'
    # bad parts  
    filename = '/app/data/store/valid/equipped not ok/BMW - S58_20211125073356_211108124552030240234182_1_NIO.png'

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


