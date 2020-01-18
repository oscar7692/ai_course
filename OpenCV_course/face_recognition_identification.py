import cv2
import numpy as np
import pickle


face_cascade = cv2.CascadeClassifier('cascades/data/haarcascade_frontalface_alt2.xml')
eyes_cascade = cv2.CascadeClassifier('cascades/data/haarcascade_eye.xml')

recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read("trainner.yml")

# open file that contains labels
labels = {"person_name": 1}
with open("labels.pickle", 'rb') as f:
    og_labels = pickle.load(f)
    labels = {v:k for k,v in og_labels.items()}


cap = cv2.VideoCapture(0)

while True:
    # capture frame by frame
    ret, frame = cap.read()
    # set fram grey scale on the frame
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.5, minNeighbors=5)
    for(x, y, w, h) in faces:
        # print(x, y, w, h) 
        # roi = regions of interest
        roi_gray = gray[y:y+h, x:x+w] # [ycord_start ycord_end, xcord_start xcord_end]
        roi_color = frame[y:y+h, x:x+w]

        # recognize? deep learned model predict keras tensorflow pytroch scikit learn
        id_, conf = recognizer.predict(roi_gray)
        if conf >= 60 and conf <= 90:
            print(id_)
            print(labels[id_])
            font = cv2.FONT_HERSHEY_SIMPLEX
            name = labels[id_]
            color = (255, 255, 255)
            stroke = 2 
            cv2.putText(frame, name, (x,y), font, 1, color, stroke, cv2.LINE_AA)

        img_item = "my_image.png"
        cv2.imwrite(img_item, roi_color)

        # display and draw a rectangle around a face in the frame
        color = (255, 0, 0) # BGR color take values from 0-255
        # how thick line gonna be
        stroke = 2
        # assigning the size of the rectangle using the coordinates + width and height
        end_cord_x = x + w
        end_cord_y = y + h
        cv2.rectangle(frame, (x , y), (end_cord_x, end_cord_y), color, stroke)
        eyes = eyes_cascade.detectMultiScale(roi_gray)
        for (ex, ey, ew, eh) in eyes:
            cv2.rectangle(roi_color, (ex, ey), (ex+ew,ey+eh), (0,255,0),2)


    # display resulting frame
    cv2.imshow('colorFrame', frame)
    if cv2.waitKey(20) & 0xFF == ord('q'):
        break

# release capture when everything is done
cap.release()
cv2.destroyAllWindows()