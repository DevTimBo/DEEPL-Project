import cv2
import os

if os.path.exists("DEEPL-Project\CAM\data\cat.mp4"):
    print("ja1")

if os.path.exists("C:\Users\hadie\Desktop\Erk Framework\DEEPL-Project\CAM\video_Frames"):
    print("ja2")

vid = cv2.VideoCapture("DEEPL-Project\CAM\data\cat.mp4")
i = 0

'''
while vid.isOpened(): 
    available, frame = vid.read()
    if available == False:
        break
    print(i)
    cv2.imwrite("DEEPL-Project\CAM\video_Frames" + str(i) + ".jpg", frame)
    i += 1

vid.release()
cv2.destroyAllWindows()

'''