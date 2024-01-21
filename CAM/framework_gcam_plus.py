import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from tensorflow.keras import Model
import matplotlib as mpl
import matplotlib.pyplot as plt
import requests
import os
import cv2
import numpy as np
from PIL import Image
from keras.utils import get_file

OUTPUT_FOLDER = "data"
result_name = "grad_cam_plusplus.jpg"


def grad_cam_plus(model, img,
                  layer_name="block5_conv3", label_name=None,
                  category_id=None):
    img_tensor = np.expand_dims(img, axis=0)

    conv_layer = model.get_layer(layer_name)
    heatmap_model = Model([model.inputs], [conv_layer.output, model.output])

    with tf.GradientTape() as gtape1:
        with tf.GradientTape() as gtape2:
            with tf.GradientTape() as gtape3:
                conv_output, predictions = heatmap_model(img_tensor)
                if category_id is None:
                    category_id = np.argmax(predictions[0])
                if label_name is not None:
                    print(label_name[category_id])
                output = predictions[:, category_id]
                conv_first_grad = gtape3.gradient(output, conv_output)
            conv_second_grad = gtape2.gradient(conv_first_grad, conv_output)
        conv_third_grad = gtape1.gradient(conv_second_grad, conv_output)

    global_sum = np.sum(conv_output, axis=(0, 1, 2))

    alpha_num = conv_second_grad[0]
    alpha_denom = conv_second_grad[0]*2.0 + conv_third_grad[0]*global_sum
    alpha_denom = np.where(alpha_denom != 0.0, alpha_denom, 1e-10)

    alphas = alpha_num/alpha_denom
    alpha_normalization_constant = np.sum(alphas, axis=(0,1))
    alphas /= alpha_normalization_constant

    weights = np.maximum(conv_first_grad[0], 0.0)

    deep_linearization_weights = np.sum(weights*alphas, axis=(0,1))
    grad_cam_map = np.sum(deep_linearization_weights*conv_output[0], axis=2)

    heatmap = np.maximum(grad_cam_map, 0)
    max_heat = np.max(heatmap)
    if max_heat == 0:
        max_heat = 1e-10
    heatmap /= max_heat

    return heatmap

def preprocess_image(img_path, target_size=(224, 224)):
    img = image.load_img(img_path, target_size=target_size)
    img = image.img_to_array(img)
    img /= 255

    return img

def show_imgwithheat(img_path, heatmap, alpha=0.4, return_array=False):
    img = cv2.imread(img_path)
    heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
    heatmap = (heatmap*255).astype("uint8")
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    superimposed_img = heatmap * alpha + img
    superimposed_img = np.clip(superimposed_img, 0, 255).astype("uint8")
    superimposed_img = cv2.cvtColor(superimposed_img, cv2.COLOR_BGR2RGB)

    imgwithheat = Image.fromarray(superimposed_img)
    try:
        #display(imgwithheat)
        plt.imshow(imgwithheat)
        plt.title("Grad-CAM++")
        plt.savefig(os.path.join(OUTPUT_FOLDER, result_name))
    except NameError:
        imgwithheat.show()

    if return_array:
        return superimposed_img

def make_gradcam_plusplus(model, img_path, last_conv_layer_name, target_size):
    # heatmap unskaliert 
    img = preprocess_image(img_path, target_size)
    heatmap_plus = grad_cam_plus(model, img, last_conv_layer_name)
    plt.imshow(heatmap_plus)
    plt.savefig(os.path.join(OUTPUT_FOLDER, result_name))
    # heatmap hochskaliert + überlagert 
    show_imgwithheat(img_path, heatmap_plus)

if __name__ == "__main__":
    import sys
    # Check if the correct number of arguments is provided
    if len(sys.argv) != 3:
        print("Usage: python tensorflow_script.py <model_name> <filepath>")
        sys.exit(1)

    # Extract command-line arguments
    model_name = sys.argv[1]
    filepath = sys.argv[2]

    if model_name == "VGG16":
        import keras.applications.vgg16 as vgg16
        # Keras Model
        model = vgg16.VGG16(weights="imagenet")
        preprocess = vgg16.preprocess_input
        decode_predictions = vgg16.decode_predictions
        last_conv_layer = "block5_conv3"
        img_size = (224, 224)
    else:
        print("SOMETHING IS WRONG!!!")
    make_gradcam_plusplus(model, filepath, last_conv_layer, img_size)
