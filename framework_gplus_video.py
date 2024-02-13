# Autor: Hadi El-Sabbagh
# Co-Autor: Tim Harmling, Jason Pranata
# Date: 13 February 2024 

# Funktionsweise:
# GradCam++ Video Funktion Integration fürs Framework

import framework_video as video
import framework_gcam_plus as gplus
import cv2
import os 
import keras 

FRAME_FOLDER = r'data\frames'
LH_frames = r'data\gcam_plus_output\Large_Heatmap'
SH_frames = r'data\gcam_plus_output\Small_Heatmap'
video_out_LH = r'data\gcam_plus_output\LH_video.avi'
video_out_SH = r'data\gcam_plus_output\SH_video.avi' 
fps = 24

OUTPUT_FOLDER = 'data\gcam_plus_output'
heatmap_name = 'cam1_1.jpg'

def make_gradcamplusplus_video(model, video_path_in, target_size, last_conv_layer_name,):
    capture = cv2.VideoCapture(video_path_in)
    video.cut_video(capture)
    sorted_frames = sorted(os.listdir(FRAME_FOLDER), key=extract_number)
    for img in sorted_frames:
        print(img)

    preds = []

    # Wendet gradcam++ auf jedes Frame an 
    i = 0
    for image in sorted_frames:
        print(image, i)
        img_path = os.path.join(FRAME_FOLDER, image)
        gplus.make_gradcam_plusplus(model, img_path, last_conv_layer_name, target_size, i)
        #preds.append(grad_cam.get_pred())
        #print(preds)
        i += 1 
        if i == 5:
            break
 
    sorted_frames_LH = sorted(os.listdir(LH_frames), key=video.extract_number)
    sorted_frames_SH = sorted(os.listdir(SH_frames), key=video.extract_number)

    video.convert_images_to_video(LH_frames, video_out_LH, fps)
    video.convert_images_to_video(SH_frames, video_out_SH, fps)

def extract_number(filename):
    filename = filename.split('.jpg')[0]
    return int(filename)


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

    make_gradcamplusplus_video(model, filepath, img_size, last_conv_layer)
    