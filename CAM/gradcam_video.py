import video_cut
import grad_cam
import cv2
import os 
import keras

video_path = 'DEEPL-Project\CAM\data\cat.mp4'
capture = cv2.VideoCapture(video_path)
FRAME_FOLDER = 'DEEPL-Project\CAM\video_Frames'
FRAME_CAM_FOLDER = ''

model_builder = keras.applications.xception.Xception
img_size = (299, 299)
preprocess_input = keras.applications.xception.preprocess_input
decode_predictions = keras.applications.xception.decode_predictions

OUTPUT_FOLDER = 'DEEPL-Project\CAM\Images\gradcam_output'
heatmap_name = 'cam1_1.jpg'
last_conv_layer_name = "block14_sepconv2_act"

model = model_builder(weights="imagenet")
model.layers[-1].activation = None  # OPTIONAL

video_cut.cut_video(capture)

for image in os.listdir(FRAME_FOLDER):
    image_path = os.path.join(FRAME_FOLDER, image)
    grad_cam.make_gradcam(model, image_path, img_size, preprocess_input, decode_predictions, last_conv_layer_name)