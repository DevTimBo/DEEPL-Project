import cv2
import os
from PIL import Image
from PIL import ImageDraw
from PIL import ImageFont
 

def cut_video(capture):
    frameNr = 0
    i = 0
    while(True): 
        success, frame = capture.read()
        if success:
            print("hallo1")
            cv2.imwrite(f'data\\frames\\{frameNr}.jpg', frame)
            print(f"Frame {frameNr} gespeichert.")
        else:
            break
        frameNr = frameNr+1
        i += 1
        if frameNr == 150:
            break
        if i == 50:
            break
    capture.release()

def get_frame_size(sample_path):
    sample_frame = cv2.imread(sample_path)
    h, w, _ = sample_frame.shape
    return (w,h)

def extract_number(filename):
    filename = filename.split('.')[0]
    filename = filename.split('_')[1]
    return int(filename)

def convert_images_to_video(Image_folder, video_path, fps):
    images = [img for img in os.listdir(Image_folder) if img.endswith(".jpg")]
    images.sort(key=extract_number)
    for img in images:
        print(img)
    frame = cv2.imread(os.path.join(Image_folder, images[0]))
    height, width, layers = frame.shape
    
    #FPS: 30
    video = cv2.VideoWriter(video_path, 0, fps, (width,height))
    for image in images:
        video.write(cv2.imread(os.path.join(Image_folder, image)))
        
    cv2.destroyAllWindows()
    video.release()

def draw_on_image(img_path, font_size, text="Der Text fehlt"):
    img = Image.open(img_path)
    I1 = ImageDraw.Draw(img)
    myFont = ImageFont.truetype('arial.ttf', font_size)
    I1.text((28, 36), text, font=myFont, fill=(255, 0, 0))
    img.save(img_path)

