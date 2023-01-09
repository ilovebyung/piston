'''
1. Create training data
2. Build autoencoder 
3. Set threshold
4. Make an inference
5. Find differences
'''

from tensorflow.keras.models import Model
from sklearn.metrics import accuracy_score, precision_score, recall_score
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.layers import Conv2D, Flatten
from tensorflow.keras.layers import Reshape, Conv2DTranspose
from tensorflow.keras.models import Model
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import plot_model
from tensorflow.keras import backend as K
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

# size = (1302/2, 6404/2)
size = (651, 3202)
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
        if image.endswith(".jpg") or image.endswith(".png"):
            try:
                data_path = pathlib.Path(data_path)
                full_name = str(pathlib.Path.joinpath(data_path, image))
                data = cv2.imread(full_name, 0)
                # resize to make sure data consistency
                resized_data = cv2.resize(data, (width, height))
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

# class Autoencoder(Model):
#     def __init__(self):
#         super(Autoencoder, self).__init__()
#         # input layer
#         self.encoder = tf.keras.Sequential([
#             layers.Flatten(),
#             layers.Conv2D(256, kernel_size=3, activation='sigmoid'),
#             layers.Conv2D(128, kernel_size=3, activation='sigmoid'),
#             layers.Conv2D(64, kernel_size=3, activation='sigmoid'),
#             layers.Conv2D(32, kernel_size=3, activation='sigmoid'),
#         ])
#         self.decoder = tf.keras.Sequential([
#             layers.Conv2D(64, kernel_size=3, activation='sigmoid'),
#             layers.Conv2D(128, kernel_size=3, activation='sigmoid'),
#             layers.Conv2D(256, kernel_size=3, activation='sigmoid'),
#             layers.Conv2D(height*width, kernel_size=3, activation='sigmoid'),
#             layers.Reshape((height, width))
#         ])

#     def call(self, x):
#         encoded = self.encoder(x)
#         decoded = self.decoder(encoded)
#         return decoded



'''
3. Set threshold
'''


def model_threshold(x_train):
    '''
    Set threshold for a model. 
    audoencoder model is a prerequisite
    '''
    encoded_imgs = autoencoder.encoder(x_train).numpy()
    decoded_imgs = autoencoder.decoder(encoded_imgs).numpy()
    loss = tf.keras.losses.mse(decoded_imgs, x_train)
    threshold = np.mean(loss) + 2*np.std(loss)
    return threshold



'''
4. Make an inference
'''


def sample_loss(filename, height, width):
    '''
    Load an image from a file and calculate sample loss
    audoencoder model is a prerequisite
    '''
    # data = np.ndarray(shape=(1, height, width), dtype=np.float32)

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
    return sample_loss

if __name__ == "__main__":

    import time
    start = time.perf_counter()

    finish = time.perf_counter()
    print(f'Finished in {round(finish-start, 2)} second(s)')

    '''
    1. Load training images
    '''


    data_path = '/home/m0034463/store/data/piston/train/equipped/'
    x_train = create_training_data(data_path, height=height, width=width)

    data_path = '/home/m0034463/store/data/piston/test/equipped/'
    x_test = create_training_data(data_path, height=height, width=width)

    '''
    2. Build autoencoder 
    '''


    # autoencoder = Autoencoder()
    # autoencoder.compile(optimizer='adam', loss=losses.MeanSquaredError())

    # reshape to (height, width, 1) and normalize input images

    x = Input(shape=(height, width,1)) 

    # Encoder
    conv1_1 = Conv2D(4, (3, 3), activation='relu', padding='same')(x)
    pool1 = MaxPooling2D((2, 2), padding='same')(conv1_1)
    conv1_2 = Conv2D(8, (3, 3), activation='relu', padding='same')(pool1)
    pool2 = MaxPooling2D((2, 2), padding='same')(conv1_2)
    conv1_3 = Conv2D(16, (3, 3), activation='relu', padding='same')(pool2)
    h = MaxPooling2D((2, 1), padding='same')(conv1_3) #<------

    # # Decoder

    conv2_1 = Conv2D(16, (3, 3), activation='relu', padding='same')(h)
    up1 = UpSampling2D((2, 2))(conv2_1)
    conv2_2 = Conv2D(8, (3, 3), activation='relu', padding='same')(up1)
    up2 = UpSampling2D((2, 2))(conv2_2)
    conv2_3 = Conv2D(4, (3, 3), activation='sigmoid', padding='same')(up2) #<--- ADD PADDING HERE
    up3 = UpSampling2D((2, 1))(conv2_3)
    r = Conv2D(1, (3, 3), activation='sigmoid', padding='same')(up3)

    model = Model(inputs=x, outputs=r)
    model.compile(optimizer='adadelta', loss='binary_crossentropy', metrics=['accuracy'])
    model.summary()


    # autoencoder = encoder + decoder
    # instantiate autoencoder model
    autoencoder = Model(inputs,
        decoder(encoder(inputs)),
        name='autoencoder')
    autoencoder.summary()

    plot_model(autoencoder,
        to_file='autoencoder.png',
        show_shapes=True)
    # Mean Square Error (MSE) loss function, Adam optimizer
    autoencoder.compile(loss='mse', optimizer='adam')

    # train the autoencoder
    autoencoder.fit(x_train,
                x_train,
                validation_data=(x_test, x_test),
                epochs=1,
                batch_size=batch_size)

    # predict the autoencoder output from test data
    x_decoded = autoencoder.predict(x_test)
    # display the 1st 8 test input and decoded images
    imgs = np.concatenate([x_test[:8], x_decoded[:8]])
    imgs = imgs.reshape((4, 4, height, width))
    imgs = np.vstack([np.hstack(i) for i in imgs])

    plt.figure()
    plt.axis('off')
    plt.title('Input: 1st 2 rows, Decoded: last 2 rows')
    plt.imshow(imgs, interpolation='none', cmap='gray')
    plt.savefig('input_and_decoded.png')
    plt.show()

    # history = autoencoder.fit(x_train, x_train,
    #                           epochs=10,
    #                           shuffle=True,
    #                           batch_size=8,
    #                           validation_data=(x_test, x_test))

    # # a summary of architecture
    # autoencoder.encoder.summary()
    # autoencoder.decoder.summary()

    # # plot history
    # plt.plot(history.history["loss"], label="Training Loss")
    # plt.plot(history.history["val_loss"], label="Validation Loss")
    # plt.legend()
    # plt.show()

    # save a mode
    autoencoder.save('./model/')

    # load autoencoder model
    if autoencoder is None:
        autoencoder = Autoencoder()
        autoencoder = keras.models.load_model('./model/')

    '''
    3. Set threshold
    '''
    threshold = model_threshold(x_train)
    # loss = tf.keras.losses.mse(decoded_imgs, x_train)
    # threshold = np.mean(loss) + np.std(loss)
    print("Loss Threshold: ", threshold)


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