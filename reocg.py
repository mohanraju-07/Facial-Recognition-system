# Advanced Video Recognition System in Python

This project demonstrates a more advanced video recognition pipeline using:

* OpenCV
* YOLOv8 (Object Detection)
* DeepSORT (Object Tracking)
* Face Recognition
* Motion Analysis
* Real-time FPS monitoring
* Video Recording

---

# Install Required Libraries

```bash
pip install ultralytics opencv-python face_recognition numpy deep-sort-realtime
```

---

# Project Structure

```text
project/
│
├── main.py
├── known_faces/
│   ├── person1.jpg
│   └── person2.jpg
└── output/
```

---

# Full Python Code

```python
import cv2
import numpy as np
import face_recognition
import os
import time
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort


# =========================
# LOAD YOLO MODEL
# =========================
model = YOLO("yolov8n.pt")


# =========================
# INITIALIZE TRACKER
# =========================
tracker = DeepSort(max_age=30)


# =========================
# LOAD KNOWN FACES
# =========================
known_face_encodings = []
known_face_names = []

face_folder = "known_faces"

for file in os.listdir(face_folder):
    path = os.path.join(face_folder, file)

    image = face_recognition.load_image_file(path)
    encodings = face_recognition.face_encodings(image)

    if len(encodings) > 0:
        known_face_encodings.append(encodings[0])
        known_face_names.append(os.path.splitext(file)[0])


# =========================
# VIDEO CAPTURE
# =========================
cap = cv2.VideoCapture(0)

frame_width = int(cap.get(3))
frame_height = int(cap.get(4))


# =========================
# VIDEO WRITER
# =========================
out = cv2.VideoWriter(
    'output/output.mp4',
    cv2.VideoWriter_fourcc(*'mp4v'),
    20,
    (frame_width, frame_height)
)


# =========================
# FPS VARIABLES
# =========================
prev_time = 0


# =========================
# MOTION DETECTION VARIABLES
# =========================
ret, previous_frame = cap.read()
previous_gray = cv2.cvtColor(previous_frame, cv2.COLOR_BGR2GRAY)


# =========================
# MAIN LOOP
# =========================
while True:
    ret, frame = cap.read()

    if not ret:
        break

    original_frame = frame.copy()


    # =========================
    # OBJECT DETECTION
    # =========================
    results = model(frame)

    detections = []

    for result in results:
        boxes = result.boxes

        for box in boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            confidence = float(box.conf[0])
            class_id = int(box.cls[0])

            label = model.names[class_id]

            if confidence > 0.5:
                detections.append(
                    ([x1, y1, x2 - x1, y2 - y1], confidence, label)
                )


    # =========================
    # OBJECT TRACKING
    # =========================
    tracks = tracker.update_tracks(detections, frame=frame)

    for track in tracks:
        if not track.is_confirmed():
            continue

        track_id = track.track_id
        ltrb = track.to_ltrb()

        x1, y1, x2, y2 = map(int, ltrb)

        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

        cv2.putText(
            frame,
            f"ID: {track_id}",
            (x1, y1 - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2
        )


    # =========================
    # FACE RECOGNITION
    # =========================
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    face_locations = face_recognition.face_locations(rgb_frame)
    face_encodings = face_recognition.face_encodings(
        rgb_frame,
        face_locations
    )

    for (top, right, bottom, left), face_encoding in zip(
        face_locations,
        face_encodings
    ):

        matches = face_recognition.compare_faces(
            known_face_encodings,
            face_encoding
        )

        name = "Unknown"

        face_distances = face_recognition.face_distance(
            known_face_encodings,
            face_encoding
        )

        if len(face_distances) > 0:
            best_match_index = np.argmin(face_distances)

            if matches[best_match_index]:
                name = known_face_names[best_match_index]


        cv2.rectangle(
            frame,
            (left, top),
            (right, bottom),
            (255, 0, 0),
            2
        )

        cv2.putText(
            frame,
            name,
            (left, top - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 0, 0),
            2
        )


    # =========================
    # MOTION DETECTION
    # =========================
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    frame_diff = cv2.absdiff(previous_gray, gray)

    _, thresh = cv2.threshold(frame_diff, 25, 255, cv2.THRESH_BINARY)

    contours, _ = cv2.findContours(
        thresh,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    motion_detected = False

    for contour in contours:
        if cv2.contourArea(contour) > 1500:
            motion_detected = True
            break

    if motion_detected:
        cv2.putText(
            frame,
            "Motion Detected",
            (20, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 0, 255),
            3
        )

    previous_gray = gray.copy()


    # =========================
    # FPS COUNTER
    # =========================
    current_time = time.time()

    fps = 1 / (current_time - prev_time)
    prev_time = current_time

    cv2.putText(
        frame,
        f"FPS: {int(fps)}",
        (20, 100),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (255, 255, 0),
        2
    )


    # =========================
    # SAVE OUTPUT VIDEO
    # =========================
    out.write(frame)


    # =========================
    # SHOW WINDOW
    # =========================
    cv2.imshow("Advanced Video Recognition", frame)


    # =========================
    # EXIT
    # =========================
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


# =========================
# RELEASE RESOURCES
# =========================
cap.release()
out.release()
cv2.destroyAllWindows()
```

---

# Features Included

## 1. YOLOv8 Object Detection

Recognizes objects like:

* Person
* Car
* Phone
* Bottle
* Dog
* Chair
* Laptop

---

## 2. DeepSORT Tracking

Tracks objects with unique IDs.

Example:

```text
Person -> ID 1
Car -> ID 2
```

Even if objects move, the same ID stays attached.

---

## 3. Face Recognition

Recognizes faces from the `known_faces` folder.

Example:

```text
known_faces/
    mohan.jpg
```

If your face appears:

```text
Mohan
```

will be shown on the screen.

---

## 4. Motion Detection

Detects movement between frames.

Useful for:

* Security systems
* CCTV
* Intrusion detection

---

## 5. FPS Monitoring

Shows real-time processing speed.

---

# How To Run

```bash
python main.py
```

Press:

```text
q
```

to quit.

---

# Upgrade Ideas

You can further improve this system with:

* Pose estimation
* Action recognition
* Emotion detection
* License plate recognition
* Weapon detection
* Crowd analysis
* AI alert system
* Database logging
* FastAPI streaming server
* Web dashboard
* GPU acceleration with CUDA

---

# For GPU Acceleration

Install CUDA version of PyTorch.

Example:

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

---

# Real-World Uses

* Smart surveillance
* Attendance systems
* AI security cameras
* Traffic monitoring
* Retail analytics
* Smart cities
* Autonomous systems
* Industrial monito
