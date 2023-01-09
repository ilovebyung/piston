'''
ONNX (Open Neural Network Exchange) is designed to be trained in one framework and then exported and used in another framework.

pip install onnx tensorflow

# nstall ONNX Runtime (ORT)
pip install onnxruntime
pip install onnxruntime-gpu
## tensorflow
pip install tf2onnx
## sklearn
pip install skl2onnx
# install from pypi
pip install -U tf2onnx
'''
import tf2onnx
import onnx
import tensorflow as tf
import tf2onnx

# Load the TensorFlow model
model = tf.saved_model.load('./model/')

# Convert the model to ONNX
onnx_model = tf2onnx.convert.from_keras(model)

# Save the ONNX model
onnx.save(onnx_model, "./model/onnx_model.onnx")

# # Use from_function for tf functions
# onnx_model, _ = tf2onnx.convert.from_keras(model, input_signature, opset=13)
# onnx.save(onnx_model, "dst/path/model.onnx")
