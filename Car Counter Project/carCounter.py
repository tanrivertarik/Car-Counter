from ultralytics import YOLO
import cv2
import cvzone
import math
from sort import *

#cap = cv2.VideoCapture(0,cv2.CAP_DSHOW)# for webscam
#cap.set(3,1280)
#cap.set(4,720)


cap = cv2.VideoCapture('cars.mp4')


model = YOLO("../Yolo-Weights/yolov8n.pt")

classnames = ["person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck", "boat",
              "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
              "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
              "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat",
              "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
              "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli",
              "carrot", "hot dog", "pizza", "donut", "cake", "chair", "sofa", "pottedplant", "bed",
              "diningtable", "toilet", "tvmonitor", "laptop", "mouse", "remote", "keyboard", "cell phone",
              "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors",
              "teddy bear", "hair drier", "toothbrush","Wallet"]
mask = cv2.imread("mask.png")

#Trackin

tracker = Sort(max_age=20,min_hits =3,iou_threshold=0.3)

while True:
    success, img = cap.read()
    imgRegion = cv2.bitwise_and(img,mask)
    results = model(imgRegion, stream=True)
    detections = np.empty((0, 5))

    for r in results:
        boxes = r.boxes
        for box in boxes:

            #kutucuklar
            x1,y1,x2,y2 = box.xyxy[0] #easier
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            #cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 255), 3) Opencv


            #cvzone
            w,h= x2-x1,y2-y1

            #confidance
            conf = math.ceil((box.conf[0]*100))/100

            cls = int(box.cls[0])

            currentClass = classnames[cls]
            #if station for only choose this objects
            if currentClass == "car" or currentClass == "truck" or currentClass =="bus" or currentClass== "motorbike" and conf >0.3:

            #Class Name
                cvzone.putTextRect(img, f'{currentClass} {conf}', (max(0, x1), max(35, y1)),
                                scale=0.5,thickness=1,offset=3)
            cvzone.cornerRect(img, (x1, y1, w, h), l=15)
            currentArray =np.array([x1,y1,x2,y2,conf])
            detections = np.vstack((detections,currentArray))
    resultsTracker = tracker.update(detections)

    for result in resultsTracker:
        x1,y1,x2,y2,Id = result
        print(result)
    cv2.imshow("Image",img)
    cv2.imshow("ImageRegion", imgRegion)
    cv2.waitKey(0)
