import cv2
from prediction import pred
import os

def highlightFace(net, frame, emotion, conf_threshold=0.7):
    frameOpencvDnn=frame.copy()

    frameHeight=frameOpencvDnn.shape[0]
    frameWidth=frameOpencvDnn.shape[1]

    blob=cv2.dnn.blobFromImage(frameOpencvDnn, 1.0, (300, 300), [104, 117, 123], True, False)

    net.setInput(blob)
    detections=net.forward()

    faceBoxes=[]
    
    for i in range(detections.shape[2]):
        
        confidence=detections[0,0,i,2]
        
        if confidence>conf_threshold:
            # формируем координаты рамки
            x1=int(detections[0,0,i,3]*frameWidth)
            y1=int(detections[0,0,i,4]*frameHeight)
            x2=int(detections[0,0,i,5]*frameWidth)
            y2=int(detections[0,0,i,6]*frameHeight)
            
            faceBoxes.append([x1,y1,x2,y2])
            # рисуем рамку на кадре
            cv2.rectangle(frameOpencvDnn, (x1,y1), (x2,y2), (0,255,0), int(round(frameHeight/150)), 8)
            cv2.putText(frameOpencvDnn, emotion, (x1, y1-15), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    
    return frameOpencvDnn,faceBoxes

faceProto=r"weights\opencv_face_detector.pbtxt"
faceModel=r"weights\opencv_face_detector_uint8.pb"

def video():

    faceNet=cv2.dnn.readNet(faceModel,faceProto)

    # получаем видео с камеры
    video=cv2.VideoCapture(0)
    while cv2.waitKey(1)<0:
        hasFrame,frame=video.read()
        
        if not hasFrame:
            
            cv2.waitKey()
            break
        filename = r"photos\video_file.jpg"
        cv2.imwrite(filename, frame)
        emotion, img = pred(filename)
        resultImg,faceBoxes=highlightFace(faceNet,frame,emotion)

        if not faceBoxes:
        # выводим в консоли, что лицо не найдено
            print("Лица не распознаны")
        
        
        
        cv2.imshow("Face detection", resultImg)
    
    if os.path.exists(filename):
        os.remove(filename)
    
        
        
video()