import cv2
import numpy as np


cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    cv2.imshow('grayFrame', gray)
    cv2.imshow('colorFrame', frame)
    
    if cv2.waitKey(20) & 0xFF == ord('q'):
        break