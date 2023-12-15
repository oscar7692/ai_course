import numpy as np
import cv2


#selecting recording device by default
cap = cv2.VideoCapture(0)


# these methods rescale the capture to a default resolution or custom one
def make_1080p():
    cap.set(3, 1920)
    cap.set(4, 1080)

def make_720p():
    cap.set(3, 1280)
    cap.set(4, 720)

def make_480p():
    cap.set(3, 640)
    cap.set(4, 480)

def change_res(width, height):
    cap.set(3, width)
    cap.set(4, height)

#this method is used to rescale the frames
def rescale_frame(frame, percent=75):
    width = int(frame.shape[1] * percent/ 100)
    height = int(frame.shape[0] * percent/ 100)
    dim = (width, height)
    return cv2.resize(frame, dim, interpolation =cv2.INTER_AREA)

while True:
    #read every frame captured by camera
    ret, frame = cap.read()
    frame = rescale_frame(frame, percent=30)
    #display current image captured on computer display
    cv2.imshow('frame', frame)
    frame2 = rescale_frame(frame, percent=140)
    cv2.imshow('frame2', frame2)
    # set wait time in 10ms and set 'q' as esc key
    if cv2.waitKey(10) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()