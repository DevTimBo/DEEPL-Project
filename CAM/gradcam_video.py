import video
import grad_cam
import cv2
import os 
import keras

#TODO Die Frame Nr. wird auch noch angezeigt, dann kann man sich dazu die anderen heatmaps anschauen 

FRAME_FOLDER = r'CAM\video_Frames'
LH_frames = r'CAM\Images\gradcam_output\Large_Heatmap'
SH_frames = r'CAM\Images\gradcam_output\Small_Heatmap'
video_out_LH = r'CAM\Images\gradcam_output\videos\LH_video.avi'
video_out_SH = r'CAM\Images\gradcam_output\videos\SH_video.avi'
fps = 24
 
OUTPUT_FOLDER = r'CAM\Images\gradcam_output'
heatmap_name = 'cam1_1.jpg'

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

 

