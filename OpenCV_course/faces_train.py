import cv2
import os
import numpy as np
from PIL import Image
import pickle


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
image_dir = os.path.join(BASE_DIR, "images")

face_cascade = cv2.CascadeClassifier('cascades/data/haarcascade_frontalface_alt2.xml')
recognizer = cv2.face.LBPHFaceRecognizer_create()

current_id = 0
label_ids = {}
y_labels = []
x_train = []

for root, dirs, files in os.walk(image_dir):
    for fl in files:
        if fl.endswith('png') or fl.endswith("jpg"):
            path = os.path.join(root, fl)
            label = os.path.basename(root).replace(" ", "-").lower()
            print(label, path)

            # creating label_ids
            if label in label_ids:
                pass
            else:
                label_ids[label] = current_id
                current_id += 1
            id_ = label_ids[label]
            # print(label_ids)
            # creating label ids dictionary
            # y_labels.append(label) # some numbers
            # x_train.append(path) # verify this image, turns into a NUMPY array, GRAY
            
            # training image to numpy array
            pil_image = Image.open(path).convert("L")
            # resize imagen for training
            size = (550, 550)
            final_image = pil_image.resize(size, Image.ANTIALIAS)
            image_array = np.array(pil_image, "uint8")
            # print(image_array)
            # region of interest "roi" in training data
            faces = face_cascade.detectMultiScale(image_array, scaleFactor=1.5, minNeighbors=5)
            
            for (x, y, w, h) in faces:
                roi = image_array[y:y+h, x:x+w]
                x_train.append(roi)
                y_labels.append(id_)
print(y_labels)
print(x_train)
# using pickle to save label ids
with open("labels.pickle", 'wb') as f:
    pickle.dump(label_ids, f)

# train the OpenCV Recognizer
recognizer.train(x_train, np.array(y_labels))
recognizer.save("trainner.yml")