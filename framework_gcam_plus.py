import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras import Model
import os
import cv2
import numpy as np
from PIL import Image

OUTPUT_FOLDER = "data"
result_name = "grad_cam_plusplus.jpg"

def grad_cam_plus(model, img,
                  layer_name="block5_conv3", label_name=None,
                  category_id=None):


    #img_tensor = np.expand_dims(img, axis=0) # commented out because input tensor shape ended up being too much by one dimension
    img_tensor = img
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
    print(f"Before Image Shape: {img.shape}")
    img = img[None]
    if model_name.strip() == "Custom":
        img = custom_model.preprocess(img)
        img = img[0]
        print(f"Image shape: {img.shape}")
    else:
        pass
    print(f"After Image Shape: {img.shape}")
    img /= 255

    return img

def show_imgwithheat(img_path, heatmap, alpha=0.4):
    img = cv2.imread(img_path)
    heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
    heatmap = (heatmap*255).astype("uint8")
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    superimposed_img = heatmap * alpha + img
    superimposed_img = np.clip(superimposed_img, 0, 255).astype("uint8")
    superimposed_img = cv2.cvtColor(superimposed_img, cv2.COLOR_BGR2RGB)

    return Image.fromarray(superimposed_img)

def make_gradcam_plusplus(model, img_path, last_conv_layer_name, target_size):
    # heatmap unskaliert 
    img = preprocess_image(img_path, target_size)
    heatmap_plus = grad_cam_plus(model, img, last_conv_layer_name)
    # heatmap hochskaliert + überlagert
    show_imgwithheat(img_path, heatmap_plus).save(os.path.join(OUTPUT_FOLDER, result_name))


if __name__ == "__main__":
    import sys

    from custom import custom_model

    # Extract command-line arguments
    model_name = sys.argv[1]
    filepath = sys.argv[2]
    last_conv_layer = sys.argv[3]
    json_string = sys.argv[4]
    import json
    # Deserialize the JSON-formatted string to get the original tuple
    img_size = json.loads(json_string)
    custom_model_path = sys.argv[5]
    custom_model_weights_path = sys.argv[6]
    print(f"Model :{model_name}:")
    if model_name == "VGG16":
        import keras.applications.vgg16 as vgg16
        # Keras Model
        model = vgg16.VGG16(weights="imagenet")
    else:
        custom_model_mapping_path = sys.argv[7]
        custom_model.set_csv_file_path(custom_model_mapping_path)
        custom_model.set_size(img_size)
        channel_num = sys.argv[8]
        custom_model.set_channels(int(channel_num))

        import keras
        model = keras.models.load_model(custom_model_path)
        model.load_weights(custom_model_weights_path)

        all_layers = model.layers
        last_conv_layer = None
        for layer in reversed(all_layers):
            if 'conv' in layer.name:
                last_conv_layer = layer.name
                break

        print(last_conv_layer)
    make_gradcam_plusplus(model, filepath, last_conv_layer, img_size)
