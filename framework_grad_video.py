from CAM import video, grad_cam
import cv2
import os 

def make_gradcam_video(model, video_path_in, img_size, preprocess_input, decode_predictions, last_conv_layer_name):
    capture = cv2.VideoCapture(video_path_in)
    # Auf 5 limitiert 
    video.cut_video(capture)
    sorted_frames = sorted(os.listdir(FRAME_FOLDER), key=extract_number)
    for img in sorted_frames:
        print(img)

    preds = []

        # Wendet gradcam auf jedes Frame an 
    i = 0
    for image in sorted_frames:
        img_path = os.path.join(FRAME_FOLDER, image)
        grad_cam.make_gradcam(model, img_path, img_size, 
                            preprocess_input, decode_predictions, last_conv_layer_name, i)
        preds.append(grad_cam.get_pred())
        print(preds)
        i += 1 
        if i == 5:
            break

    sorted_frames_LH = sorted(os.listdir(LH_frames), key=video.extract_number)
    sorted_frames_SH = sorted(os.listdir(SH_frames), key=video.extract_number)

    # Fügt jedem Bild aus dem LH Ordner Text hinzu 
    i = 0
    for index, img in enumerate(sorted_frames_LH):
        img_path = os.path.join(LH_frames, img)
        video.draw_on_image(img_path, 20, str(preds[index]))
        i += 1 
        if i == 5:
            break

    # Fügt jedem Bild aus dem SH Ordner Text hinzu
    i = 0
    for index, img in enumerate(sorted_frames_SH):
        img_path = os.path.join(SH_frames, img)
        video.draw_on_image(img_path, 20, str(preds[index]))
        i += 1 
        if i == 5:
            break
    
    # teste 
    #video_Frames = r'CAM\video_Frames'
    video.convert_images_to_video(LH_frames, video_out_LH, fps)
    video.convert_images_to_video(SH_frames, video_out_SH, fps)

def extract_number(filename):
    filename = filename.split('.jpg')[0]
    return int(filename)

if __name__ == "__main__":
    import sys
    from custom import custom_model
    
    FRAME_FOLDER = r'CAM\video_Frames'
    LH_frames = r'CAM\Images\gradcam_output\Large_Heatmap'
    SH_frames = r'CAM\Images\gradcam_output\Small_Heatmap'
    video_out_LH = r'data\LH_video.avi'
    video_out_SH = r'data\SH_video.avi' 
    fps = 24
    
    # Extract command-line arguments
    model_name = sys.argv[1]
    filepath = sys.argv[2]
    last_conv_layer = sys.argv[3]
    json_string = sys.argv[4]
    import json
    
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
    preprocess = custom_model.preprocess
    decode_predictions = custom_model.decode_predictions

    # Call your function
    make_gradcam_video(model, filepath, img_size, preprocess,
                       decode_predictions, last_conv_layer)
    
    