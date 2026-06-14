from ultralytics import YOLO
import cv2 as cv 
from keras_facenet import FaceNet 
import numpy as np 
import os 
import pickle 
from numpy.linalg import norm


model = YOLO('yolov8n-face.pt')
embedder = FaceNet()

DATASET ="dataset"

known_labels=[]
known_embeddings =[] 

for person in os.listdir(DATASET):
    person_path = os.path.join(DATASET,person)

    for img_name in os.listdir(person_path):
        img_path = os.path.join(person_path,img_name)

        img = cv.imread(img_path)
        result = model(img)

        box = result[0].boxes[0]
        if len(result[0].boxes) == 0:
            print("No face found:", img_path)
            continue
            
        x1, y1, x2, y2 = map(int, box.xyxy[0])

        faceimg = result[0].orig_img[y1:y2, x1:x2]

        img = cv.resize(faceimg,(160,160))

        embeddings = embedder.embeddings([img])[0]

        known_embeddings.append(embeddings)
        known_labels.append(person)


with open("faces_db.pkl","wb") as f :
    pickle.dump((known_embeddings,known_labels),f)

print("Total faces saved:", len(known_embeddings))
