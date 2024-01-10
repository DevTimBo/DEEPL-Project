import cv2

vid = cv2.VideoCapture('')
i = 0

while vid.isOpened(): 
    available, frame = vid.read()
    if available == False:
        break
    cv2.imwrite("DEEPL-Project\CAM\video_Frames" + str(i) + ".jpg", frame)