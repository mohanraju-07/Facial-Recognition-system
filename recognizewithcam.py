from ultralytics import YOLO
import cv2 as cv
from keras_facenet import FaceNet
import pickle
from numpy.linalg import norm
import cvzone

def distance(a,b):
    return norm(a-b)

model = YOLO('yolov8n-face.pt')
embedder = FaceNet()

# Load saved faces ONCE
with open("faces_db.pkl","rb") as f:
    known_embeddings, known_labels = pickle.load(f)

cap = cv.VideoCapture(0)

while True:
    success, img = cap.read()

    rslt = model(img)

    for box in rslt[0].boxes:

        x1,y1,x2,y2 = map(int, box.xyxy[0])

        bbox = [x1, y1, x2-x1, y2-y1]
        cvzone.cornerRect(img, bbox, l=9, rt=4)

        faceimg = img[y1:y2, x1:x2]
        tstimg = cv.resize(faceimg, (160,160))

        emb = embedder.embeddings([tstimg])[0]

        min_dist = float('inf')
        index = -1

        for i in range(len(known_embeddings)):
            dst = distance(emb, known_embeddings[i])
            if dst < min_dist:
                min_dist = dst
                index = i

        # name = known_labels[index]
        if index != -1 and min_dist < 0.9:
            name = known_labels[index]
        else:
            name = "Unknown"

        cvzone.putTextRect(img, name, (x1, y1-10))

    cv.imshow("video", img)

    if cv.waitKey(20) & 0xff == ord('d'):
        break

cap.release()
cv.destroyAllWindows()
