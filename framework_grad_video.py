# Autor: Hadi El-Sabbagh
# Co-Autor: Tim Harmling, Jason Pranata
# Date: 13 February 2024 

# Beschreibung: Die make_gradcam_video Funktion ist das Bindeglied zwischen der video.py und grad_cam.py
# Hier wird die Videofunktion auf Grad-Cam angewendet 

# Funktionsweise:
# GradCam Video Funktion Integration fürs Framework

import framework_video as video
import cv2
import os 
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import keras
import tensorflow as tf

FRAME_FOLDER = r'data\frames'
LH_frames = r'data\gradcam_output\Large_Heatmap'
SH_frames = r'data\gradcam_output\Small_Heatmap'
video_out_LH = r'data\gradcam_output\LH_video.avi'
video_out_SH = r'data\gradcam_output\SH_video.avi' 
fps = 24

def make_gradcam_video(model, video_path_in, img_size, preprocess_input, decode_predictions, last_conv_layer_name):
    capture = cv2.VideoCapture(video_path_in)
    video.cut_video(capture)
    sorted_frames = sorted(os.listdir(FRAME_FOLDER), key=extract_number)
    for img in sorted_frames:
        print(img)

    preds = []

        # Wendet gradcam auf jedes Frame an 
    i = 0
    for image in sorted_frames:
        img_path = os.path.join(FRAME_FOLDER, image)
        make_gradcam(model, img_path, img_size, 
                            preprocess_input, decode_predictions, last_conv_layer_name, i)
        preds.append(get_pred())
        print(preds)
        i += 1 
        if i == 100:
            break

    sorted_frames_LH = sorted(os.listdir(LH_frames), key=video.extract_number)
    sorted_frames_SH = sorted(os.listdir(SH_frames), key=video.extract_number)

    # Fügt jedem Bild aus dem LH Ordner Text hinzu 
    i = 0
    for index, img in enumerate(sorted_frames_LH):
        img_path = os.path.join(LH_frames, img)
        video.draw_on_image(img_path, 20, str(preds[index]))
        i += 1 
        if i == 100:
            break

    # Fügt jedem Bild aus dem SH Ordner Text hinzu
    i = 0
    for index, img in enumerate(sorted_frames_SH):
        img_path = os.path.join(SH_frames, img)
        video.draw_on_image(img_path, 20, str(preds[index]))
        i += 1 
        if i == 100:
            break
    
    # teste 
    #video_Frames = r'CAM\video_Frames'
    video.convert_images_to_video(LH_frames, video_out_LH, fps)
    video.convert_images_to_video(SH_frames, video_out_SH, fps)

def extract_number(filename):
    filename = filename.split('.jpg')[0]
    return int(filename)

OUTPUT_FOLDER = 'data\gradcam_output'
OUTPUT_FOLDER_SH = r'data\gradcam_output\Small_Heatmap'
OUTPUT_FOLDER_MH = r'data\gradcam_output\Mid_Heatmap'
OUTPUT_FOLDER_LH = r'data\gradcam_output\Large_Heatmap'

def make_gradcam(model, img_path, img_size, preprocess, decode_predictions, last_conv_layer_name, frameNr = ''):
    global prediction, heatmap_name2
    heatmap_name = f'cam1_1{frameNr}.jpg'
    heatmap_name2 = f'cam1_2{frameNr}.jpg'
    result_name = f'cam1_3{frameNr}.jpg'
    preprocess_input = preprocess
    img_array = preprocess_input(get_img_array(img_path, size=img_size))
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
    global superimposed_img, test_superim

    img = keras.utils.load_img(img_path)
    img = keras.utils.img_to_array(img)

    heatmap = np.uint8(255 * heatmap)

    jet = mpl.colormaps["jet"]

    jet_colors = jet(np.arange(256))[:, :3]
    jet_heatmap = jet_colors[heatmap]

    jet_heatmap = keras.utils.array_to_img(jet_heatmap)
    jet_heatmap = jet_heatmap.resize((img.shape[1], img.shape[0]))
    jet_heatmap = keras.utils.img_to_array(jet_heatmap)

    superimposed_img = jet_heatmap * alpha
    superimposed_img = keras.utils.array_to_img(superimposed_img)
    mid_path = os.path.join(OUTPUT_FOLDER_MH, heatmap_name2)
    superimposed_img.save(mid_path)

    superimposed_img = jet_heatmap * alpha + img
    superimposed_img = keras.utils.array_to_img(superimposed_img)

    superimposed_img.save(cam_path)

    # display(Image(cam_path)) --> Original
    # plt.imshow(superimposed_img)
    # plt.colorbar()
    # plt.title(decode_predictions(preds, top=1)[0])

def get_pred():
    return prediction

if __name__ == "__main__":
    import sys
    from custom import custom_model
    
    # Extract command-line arguments
    model_name = sys.argv[1]
    filepath = sys.argv[2]
    json_string = sys.argv[3]
    import json
    
    img_size = json.loads(json_string)
    custom_model_path = sys.argv[4]
    custom_model_weights_path = sys.argv[5]
    print(f"Model :{model_name}:")

    if model_name.strip() == "VGG16":
        import keras.applications.vgg16 as vgg16

        # Keras Model
        model = vgg16.VGG16(weights="imagenet")
        img_size = (224, 224)
        preprocess = vgg16.preprocess_input
        decode_predictions = vgg16.decode_predictions
    elif model_name.strip() == "VGG19":
        import keras.applications.vgg19 as VGG19

        # Keras Model
        model = VGG19.VGG19(weights="imagenet")
        img_size = (224, 224)
        preprocess = VGG19.preprocess_input
        decode_predictions = VGG19.decode_predictions
    elif model_name.strip() == "ResNet50":
        import keras.applications.resnet50 as ResNet50

        # Keras Model
        model = ResNet50.ResNet50(weights="imagenet")
        img_size = (224, 224)
        preprocess = ResNet50.preprocess_input
        decode_predictions = ResNet50.decode_predictions
    else:

        custom_model_mapping_path = sys.argv[6]
        custom_model.set_csv_file_path(custom_model_mapping_path)
        custom_model.set_size(img_size)
        channel_num = sys.argv[7]
        custom_model.set_channels(int(channel_num))

        import keras

        model = keras.models.load_model(custom_model_path)
        model.load_weights(custom_model_weights_path)
        preprocess = custom_model.preprocess
        decode_predictions = custom_model.decode_predictions

    all_layers = model.layers
    last_conv_layer = None
    for layer in reversed(all_layers):
        if 'conv' in layer.name:
            last_conv_layer = layer.name
            break

    make_gradcam_video(model, filepath, img_size, preprocess,
                       decode_predictions, last_conv_layer)
    
    