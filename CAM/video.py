# Autor: Hadi El-Sabbagh 
# Date: 13 February 2024 
# Beschreibung: Dieses Skript definiert Methoden zur Nutzung der Videofunktionen

import cv2
import os
from PIL import Image
from PIL import ImageDraw
from PIL import ImageFont

# Extrahiert Frames aus einem Video und speichert sie als einzelne Bilder 
def cut_video(capture):
    frameNr = 0
    i = 0
    while(True): 
        success, frame = capture.read()
        if success:
            cv2.imwrite(f'C:/Users/hadie/Desktop/Erk Framework/DEEPL-Project/CAM/video_Frames/{frameNr}.jpg', frame)
            print(f"Frame {frameNr} gespeichert.")
        else:
            break
        frameNr = frameNr+1
        i += 1
        if frameNr == 150: ## BEGRENZUNG!!! --> Damit beim Testen nicht direkt das GANZE Video 
            break
        #if i == 5:
            #break
    capture.release()

# Ermittelt die Größe (Breite und Höhe) eines Bildes
def get_frame_size(sample_path):
    sample_frame = cv2.imread(sample_path)
    h, w, _ = sample_frame.shape
    return (w,h)

# Wird genutzt um die Frames zu sortieren, extrahiert die Nummer aus dem Dateinamen eines Bildes
def extract_number(filename):
    filename = filename.split('.')[0]
    filename = filename.split('_')[1]
    return int(filename)

# Konvertiert eine Reihe von Bildern zu einem Video
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

# Fügt einem Bild Text hinzu, also die Klasse 
def draw_on_image(img_path, font_size, text="Der Text fehlt"):
    img = Image.open(img_path)
    I1 = ImageDraw.Draw(img)
    myFont = ImageFont.truetype('arial.ttf', font_size)
    I1.text((28, 36), text, font=myFont, fill=(255, 0, 0))
    img.save(img_path)

