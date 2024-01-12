# Imports
import keras.applications.vgg16 as vgg16
import cv2 as cv
import tensorflow as tf

import LRP
import PLOTTING
import IMAGE_EDIT

tf.compat.v1.disable_eager_execution()

# Keras Model
model = vgg16.VGG16(weights="imagenet")
preprocess = vgg16.preprocess_input
decode_predictions = vgg16.decode_predictions


def predictions_model(image, model):
    x = preprocess(image[None])
    predictions = model.predict(x)
    decoded_predictions = decode_predictions(predictions, top=5)[0]
    return decoded_predictions[0]




# Load Image
image_path = "../data/tabby-cat.png"
image = cv.imread(image_path)
image = cv.resize(image, [224, 224])

# Config

# Plotting N Images
lrp_image = LRP.analyze_image_lrp(image, model, preprocess,"lrp.alpha_1_beta_0")
titles = ["Original Image", "LRP Image", "LRP Image"]
images = [image, lrp_image, lrp_image]
cmaps = ["gray", "gray", "gray"]
PLOTTING.plot_n_images(images, titles, cmaps, figsize=(5, 3))
#
# # Time Plot
# cmap = "gray"
# figsize = (15, 5)
# PLOTTING.time_plot_with_avg(cmap, figsize,
#                             lrp_image, lrp_image, lrp_image)
#
# # Noise Plot
# noise_images = []
# noise_lrp_images = []
# noise_level = 25
# for i in range(5):
#     noise_image = IMAGE_EDIT.add_noise(image, noise_level=50)
#     noise_images.append(noise_image)
#     noise_level = noise_level + 50
#     noise_lrp_images.append(LRP.analyze_image_lrp(noise_image, model, preprocess))
#
# PLOTTING.plot_originals_and_noises(analyzed_images=noise_lrp_images,
#                                    noise_images=noise_images, title="LRP Noise Analyzer")
#
# # Overlap Plot
# images = [image, noise_images, lrp_image]
# PLOTTING.overlap_and_plot(images=images)

