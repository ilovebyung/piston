from tensorflow.keras.utils import plot_model
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras import backend as K
import tensorflow.keras as keras
import numpy as np
import os
import cv2


'''
Load corresponding model
'''

path = '/home/m0034463/store/data/piston/d0'
# path = '/home/m0034463/store/data/piston/d3'

os.chdir(path)
files = os.listdir()
image = cv2.imread(files[0],0)
size = image.shape
height= size[0]
width = size[1]
input_image_shape = (height,width,1)


# load model
os.chdir(path)
model = keras.models.load_model('./model/')
# d3 = keras.models.load_model('./model/')

print(model.summary())

def get_model_memory_usage(batch_size, model):
    
    features_mem = 0 # Initialize memory for features. 
    float_bytes = 4.0 #Multiplication factor as all values we store would be float32.
    
    for layer in model.layers:

        out_shape = layer.output_shape
        
        if type(out_shape) is list:   #e.g. input layer which is a list
            out_shape = out_shape[0]
        else:
            # out_shape = [out_shape[1], out_shape[2], out_shape[3]]
            # out_shape = [out_shape[1], out_shape[2]]
            out_shape = out_shape[2]
            
        #Multiply all shapes to get the total number per layer.    
        single_layer_mem = 1 
        for s in out_shape:
            if s is None:
                continue
            single_layer_mem *= s
        
        single_layer_mem_float = single_layer_mem * float_bytes #Multiply by 4 bytes (float)
        single_layer_mem_MB = single_layer_mem_float/(1024**2)  #Convert to MB
        
        print("Memory for", out_shape, " layer in MB is:", single_layer_mem_MB)
        features_mem += single_layer_mem_MB  #Add to total feature memory count

# Calculate Parameter memory
    trainable_wts = np.sum([K.count_params(p) for p in model.trainable_weights])
    non_trainable_wts = np.sum([K.count_params(p) for p in model.non_trainable_weights])
    parameter_mem_MB = ((trainable_wts + non_trainable_wts) * float_bytes)/(1024**2)
    print("_________________________________________")
    print("Memory for features in MB is:", features_mem*batch_size)
    print("Memory for parameters in MB is: %.2f" %parameter_mem_MB)

    total_memory_MB = (batch_size * features_mem) + parameter_mem_MB  #Same number of parameters. independent of batch size

    total_memory_GB = total_memory_MB/1024
    
    return total_memory_GB

#####################################################################

mem_for_my_model = get_model_memory_usage(1, model)

print("_________________________________________")
print("Minimum memory required to work with this model is: %.2f" %mem_for_my_model, "GB")


###############################################################

model.summary()