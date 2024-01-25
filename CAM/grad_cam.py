import os

os.environ["KERAS_BACKEND"] = "tensorflow"
import numpy as np
import tensorflow as tf
import keras
from IPython.display import Image, display
import matplotlib as mpl
import matplotlib.pyplot as plt

prediction = None

'''
# Für dem Testdurchlauf
model_builder = keras.applications.xception.Xception
img_size = (299, 299)
preprocess_input = keras.applications.xception.preprocess_input
decode_predictions = keras.applications.xception.decode_predictions

OUTPUT_FOLDER = 'DEEPL-Project\CAM\Images\gradcam_output'
heatmap_name = 'cam1_1.jpg'
result_name = 'cam1_2.jpg'
last_conv_layer_name = "block14_sepconv2_act"

model = model_builder(weights="imagenet")
model.layers[-1].activation = None  # OPTIONAL


img_path = keras.utils.get_file(
    "hund.jpg",
   "https://einfachtierisch.de/media/cache/article_main_image_tablet/cms/2013/05/Hundewelpe-Retriever-Halsband.jpg?522506"
)
'''

def make_gradcam(model, img_path, img_size, preprocess, decode_predictions, last_conv_layer_name, frameNr = ''):
    global prediction 
    OUTPUT_FOLDER_SH = r'CAM\Images\gradcam_output\Small_Heatmap'
    OUTPUT_FOLDER_LH = r'CAM\Images\gradcam_output\Large_Heatmap'
    heatmap_name = f'cam1_1{frameNr}.jpg'
    result_name = f'cam1_2{frameNr}.jpg'
    preprocess_input = preprocess
    img_array = preprocess_input(get_img_array(img_path, size=img_size))
    # print(img_array)
    model.layers[-1].activation = None  # OPTIONAL
    preds = model.predict(img_array)
    prediction = decode_predictions(preds, top=1)[0]
    print("Predicted:", prediction)
    heatmap = make_gradcam_heatmap(img_array, model, last_conv_layer_name)
    plt.imshow(heatmap)
    plt.savefig(os.path.join(OUTPUT_FOLDER_SH, heatmap_name))
    result_path = os.path.join(OUTPUT_FOLDER_LH, result_name)
    save_and_display_gradcam(img_path, preds, heatmap, result_path, alpha=0.4)



def get_img_array(img_path, size):
    img = keras.utils.load_img(img_path, target_size=size)
    array = keras.utils.img_to_array(img)
    array = np.expand_dims(array, axis=0)
    return array

def make_gradcam_heatmap(img_array, model, last_conv_layer_name, pred_index=None):
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


def save_and_display_gradcam(img_path, preds, heatmap, cam_path, alpha=0.4):
    img = keras.utils.load_img(img_path)
    img = keras.utils.img_to_array(img)

    heatmap = np.uint8(255 * heatmap)

    jet = mpl.colormaps["jet"]

    jet_colors = jet(np.arange(256))[:, :3]
    jet_heatmap = jet_colors[heatmap]

    jet_heatmap = keras.utils.array_to_img(jet_heatmap)
    jet_heatmap = jet_heatmap.resize((img.shape[1], img.shape[0]))
    jet_heatmap = keras.utils.img_to_array(jet_heatmap)

    superimposed_img = jet_heatmap * alpha + img
    superimposed_img = keras.utils.array_to_img(superimposed_img)

    superimposed_img.save(cam_path)

    # display(Image(cam_path)) --> Original
    # plt.imshow(superimposed_img)
    # plt.colorbar()
    # plt.title(decode_predictions(preds, top=1)[0])

def get_pred():
    return prediction
#make_gradcam(model, img_path, img_size, preprocess_input, decode_predictions, last_conv_layer_name)