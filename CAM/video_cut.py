import cv2
import os

video_path = 'DEEPL-Project\CAM\data\cat.mp4'
capture = cv2.VideoCapture(video_path)

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

cut_video(capture)
