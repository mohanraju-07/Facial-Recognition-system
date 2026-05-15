# Facial Recognition System

**README.md**
================================

# Project Overview
---------------

The Facial Recognition System is a Python-based project that utilizes machine learning models to identify individuals based on their facial features. The system consists of two main scripts: `embeddings.py` and `recognizewithcam.py`. These scripts leverage the YOLOv8 (You Only Look Once version 8) and FaceNet models to create a robust facial recognition system.

# Folder Structure & Explanation
-------------------------------

The project is organized into a single folder with two Python scripts:
```markdown
.
embeddings.py
recognizewithcam.py
README.md
```
Here's a brief explanation of each script:

### embeddings.py

This script is used to create a database of known faces. It utilizes the YOLOv8 model to detect faces in images and extracts feature embeddings from these faces using FaceNet. The extracted embeddings are stored in a database for future reference.

### recognizewithcam.py

This script uses the previously created database to recognize faces in real-time from a camera feed. It integrates the YOLOv8 model to detect faces and matches the extracted feature embeddings against the known faces in the database using FaceNet.

# Features
-----

The Facial Recognition System offers the following features:

*   **Face Detection**: Utilizes YOLOv8 to detect faces in images or videos
*   **Face Embeddings**: Uses FaceNet to extract feature embeddings from detected faces
*   **Face Recognition**: Compares extracted feature embeddings against a database of known faces
*   **Real-time Recognition**: Ability to recognize faces in real-time from a camera feed

# Technologies Used
------------------

The project employs the following technologies:

*   **Python 3.x**: Programming language used for scripting
*   **YOLOv8**: Deep learning model for face detection
*   **FaceNet**: Deep learning model for face recognition and feature embedding
*   **OpenCV**: Computer vision library for image and video processing
*   **TensorFlow/Keras**: Deep learning framework for model deployment

# How to Run the Project
------------------------

To run the project, follow these steps:

### Prerequisites

*   Install the required dependencies using pip:
    ```bash
pip install opencv-python tensorflow keras
```
*   Install the YOLOv8 model using pip:
    ```bash
pip install ultralytics
```
*   Clone this project repository.

### Running the Scripts

1.  Create a database of known faces by running `embeddings.py` with a set of images containing the faces to be recognized.
    ```bash
python embeddings.py --image_path=/path/to/images
```
2.  Run `recognizewithcam.py` to recognize faces in real-time from a camera feed.
    ```bash
python recognizewithcam.py
```

This will initiate the facial recognition system and display the recognized faces in real-time. Make sure to replace `/path/to/images` with the actual path to the images used to create the database.