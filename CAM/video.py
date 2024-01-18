import cv2
import os

#Für einen Testdurchlauf 
#video_path = 'DEEPL-Project\CAM\data\cat.mp4'
#capture = cv2.VideoCapture(video_path)

def cut_video(capture):

    frameNr = 0

    while(True): 
        success, frame = capture.read()

        if success:
            print("hallo1")
            cv2.imwrite(f'C:/Users/hadie/Desktop/Erk Framework/DEEPL-Project/CAM/video_Frames/{frameNr}.jpg', frame)
            print(f"Frame {frameNr} gespeichert.")
        
        else:
            break

        frameNr = frameNr+1
        
        if frameNr == 2:
            break
    
    capture.release()

def get_frame_size(sample_path):
    sample_frame = cv2.imread(sample_path)
    h, w, _ = sample_frame.shape
    return (w,h)

def convert_images_to_video(Image_folder, video_path):
    images = [img for img in os.listdir(Image_folder) if img.endswith(".jpg")]
    frame = cv2.imread(os.path.join(Image_folder, images[0]))
    height, width, layers = frame.shape
    
    video = cv2.VideoWriter(video_path, 0, 1, (width,height))
    for image in images:
        video.write(cv2.imread(os.path.join(Image_folder, video_path)))
        
    cv2.destroyAllWindows()
    video.release()