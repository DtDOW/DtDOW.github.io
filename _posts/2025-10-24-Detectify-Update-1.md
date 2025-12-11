---
title: Detectify Update 1 
date: 2025-10-24
categories: [project, closed]
tags: [project, closed, technology, detectify]
---

## Detectify - Scanning Faces : First step towards detecting deepfakes. 

The first step in detecting deepfakes is to scan the face and then extract features from it (we’ll talk about that later).

For scanning faces, we use OpenCV and MediaPipe. I’ve included my code below:

```python 
import cv2 as cv 
import mediapipe as mp 


mp_face_detection = mp.solutions.face_detection



webcam = cv.VideoCapture(0)

with mp_face_detection.FaceDetection(min_detection_confidence = 0.5, model_selection = 1) as face_detection :
    
    while True : 
        ret, frame = webcam.read()
        cv.imshow('frame', frame)  
        img_rgb = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
        out = face_detection.process(img_rgb)
        #print(out.detections)
        H, W, _ = frame.shape
        if out.detections is not None : 
            for i in out.detections : 
                location_data = i.location_data 
                box = i.location_data.relative_bounding_box
                x,y,w,h = box.xmin, box.ymin, box.width, box.height

                x = int(x*W)
                y = int(y*H)
                w = int(w*W)
                h = int(h*H)

                cv.rectangle(frame, (x,y), (x+w, y+h), (0,0,255), 10)
        cv.imshow('frame', frame)              
        if cv.waitKey(1) & 0xFF == ord('q') : 
            break 


webcam.release() 
cv.destroyAllWindows()

``` 
## Video Example : 

Here I am pasting a link to working video example where I am scanning my face. 

https://drive.google.com/file/d/1qTUCVU1qj25KikMadW7pSmR1XE7JfpE3/view?usp=sharing