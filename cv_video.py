import cv2
from prediction import pred


def video():

    # получаем видео с камеры
    video=cv2.VideoCapture(0)
    while cv2.waitKey(1)<0:
        hasFrame,frame=video.read()
        
        if not hasFrame:
            
            cv2.waitKey()
            break
        
        filename = r"photos\Face_video.jpg"
        cv2.imshow("Face detection", frame)
        cv2.imwrite(filename, frame)
        pred(filename)
        

video()