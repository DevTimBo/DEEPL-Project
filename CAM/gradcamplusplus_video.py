import video
import grad_cam_plusplus
from grad_cam_plusplus import make_gradcam_plusplus
import cv2
import os 
import keras 
import time

FRAME_FOLDER = r'CAM\video_Frames'
LH_frames = r'CAM\Images\gradcamplusplus_output\Large_Heatmap'
SH_frames = r'CAM\Images\gradcamplusplus_output\Mid_Heatmap'
video_out_LH = r'CAM\Images\gradcamplusplus_output\videos\LH_video.avi'
video_out_SH = r'CAM\Images\gradcamplusplus_output\videos\SH_video.avi'
fps = 24

OUTPUT_FOLDER = r'CAM\Images\gradcamplusplus_output'
heatmap_name = 'cam1_1.jpg'

def make_gradcamplusplus_video(model, video_path_in, target_size, last_conv_layer_name,):
    start_time = time.time()  
    capture = cv2.VideoCapture(video_path_in)
    # Auf 5 limitiert 
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
        make_gradcam_plusplus(model, img_path, last_conv_layer_name, target_size, i)
        #preds.append(grad_cam.get_pred())
        #print(preds)
        i += 1 
        #if i == 5:
            #break
 
    sorted_frames_LH = sorted(os.listdir(LH_frames), key=video.extract_number)
    sorted_frames_SH = sorted(os.listdir(SH_frames), key=video.extract_number)

    video.convert_images_to_video(LH_frames, video_out_LH, fps)
    video.convert_images_to_video(SH_frames, video_out_SH, fps)

    end_time = time.time() 
    total_time = end_time - start_time 
    print(f"Total execution time(Video): {total_time} seconds")

"""     # Fügt jedem Bild aus dem LH Ordner Text hinzu 
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
            break """

def extract_number(filename):
    filename = filename.split('.jpg')[0]
    return int(filename)

    