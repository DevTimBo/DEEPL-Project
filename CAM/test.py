## Ist dafür gedacht die Module zu testen 

from grad_cam import make_gradcam
from grad_cam_plusplus import make_gradcam_plusplus
from grad_cam_plusplus import vgg16_mura_model
from gradcam_video import make_gradcam_video
from gradcamplusplus_video import make_gradcamplusplus_video
import os
os.environ["KERAS_BACKEND"] = "tensorflow"
import numpy as np
import tensorflow as tf
import keras
from IPython.display import Image, display
import matplotlib as mpl
import matplotlib.pyplot as plt
from PIL import Image


# Für Grad-CAM 
""" model_builder = keras.applications.xception.Xception
img_size = (299, 299)
preprocess_input = keras.applications.xception.preprocess_input
decode_predictions = keras.applications.xception.decode_predictions

heatmap_name = 'cam1_1.jpg'
result_name = 'cam1_2.jpg'
last_conv_layer_name = "block14_sepconv2_act"

model = model_builder(weights="imagenet")
model.layers[-1].activation = None  # OPTIONAL  """

#img_path = keras.utils.get_file(
#    "hund.jpg",
##   "https://einfachtierisch.de/media/cache/article_main_image_tablet/cms/2013/05/Hundewelpe-Retriever-Halsband.jpg?522506"
#)  

# + Video 

video_path_in = r'CAM\data\cat.mp4'

#------------------------------------------

# Für Grad-CAM++
WEIGHTS_PATH_VGG16_MURA = "https://github.com/samson6460/tf_keras_gradcamplusplus/releases/download/Weights/tf_keras_vgg16_mura_model.h5"
#TODO Hier auch Name konstant halten 
last_conv_layer_name = "block5_conv3"
target_size = (224, 224)
model = vgg16_mura_model() 

img_path = r'CAM\Images\puppy.jpg'

#make_gradcam(model, img_path, img_size, preprocess_input, decode_predictions, last_conv_layer_name)
make_gradcam_plusplus(model, img_path, last_conv_layer_name, target_size)
#make_gradcam_video(model, video_path_in, img_size, preprocess_input, decode_predictions, last_conv_layer_name)
#make_gradcamplusplus_video(model, video_path_in, target_size, last_conv_layer_name)