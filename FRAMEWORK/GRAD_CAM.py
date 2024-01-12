import os

os.environ["KERAS_BACKEND"] = "tensorflow"

import numpy as np
import tensorflow as tf
import keras

from IPython.display import Image, display
import matplotlib as mpl
import matplotlib.pyplot as plt

def make_grad_cam(model, img_path, img_size, preprocess, decode_predictions, last_conv_layer_name):
    preprocess_input = preprocess
    img_array = preprocess_input(get_img_array(img_path, size=img_size))
    print(img_array)
    model.layers[-1].activation = None # OPTIONAL
    preds = model.predict(img_array)
    print("Predicted:", decode_predictions(preds, top=1)[0])
    heatmap = make_gradcam_heatmap(img_array, model, last_conv_layer_name)
    return heatmap

def get_img_array(img_path, size):
    img = keras.utils.load_img(img_path, target_size=size)
    array = keras.utils.img_to_array(img)
    array = np.expand_dims(array, axis=0)
    return array

def make_gradcam_heatmap(img_array, model, last_conv_layer_name, pred_index=None):
    import tensorflow as tf
    tf.compat.v1.enable_eager_execution()

    grad_model = keras.models.Model(
        model.inputs, [model.get_layer(last_conv_layer_name).output, model.output]
    )

    with tf.GradientTape() as tape:
        last_conv_layer_output, preds = grad_model(img_array)
        if pred_index is None:
            pred_index = tf.argmax(preds[0])
        class_channel = preds[:, pred_index]

    grads = tape.gradient(class_channel, last_conv_layer_output)

    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    last_conv_layer_output = last_conv_layer_output[0]
    heatmap = last_conv_layer_output @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)

    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
    return heatmap.numpy()