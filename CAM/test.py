import grad_cam
import os
os.environ["KERAS_BACKEND"] = "tensorflow"
import numpy as np
import tensorflow as tf
import keras
from IPython.display import Image, display
import matplotlib as mpl
import matplotlib.pyplot as plt
from PIL import Image

model_builder = keras.applications.xception.Xception
img_size = (299, 299)
preprocess_input = keras.applications.xception.preprocess_input
decode_predictions = keras.applications.xception.decode_predictions

heatmap_name = 'cam1_1.jpg'
result_name = 'cam1_2.jpg'
last_conv_layer_name = "block14_sepconv2_act"

model = model_builder(weights="imagenet")
model.layers[-1].activation = None  # OPTIONAL


img_path = keras.utils.get_file(
    "hund.jpg",
   "https://einfachtierisch.de/media/cache/article_main_image_tablet/cms/2013/05/Hundewelpe-Retriever-Halsband.jpg?522506"
)

grad_cam.make_gradcam(model, img_path, img_size, preprocess_input, decode_predictions, last_conv_layer_name)