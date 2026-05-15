**Facial-Recognition-System**
==========================

**Project Overview**
-------------------

Welcome to the Facial-Recognition-System, a Python-based project designed to create a database of face embeddings and recognize faces in real-time using a webcam feed. This project utilizes deep learning techniques to build a comprehensive facial recognition system.

**Folder Structure & Explanation**
---------------------------------

```bash
face_recognition/
embeddings.py    # Script to create a database of face embeddings
recognizewithcam.py  # Script to recognize faces in real-time using a webcam feed
dataset/      # Directory to store face embeddings database
faces_db.pkl     # Pickled database of face embeddings
requirements.txt # List of project dependencies
README.md     # Project documentation (this file)
```

*   `.gitignore` has been omitted from the directory structure, as it should not be included in the root of the project. It should be placed within the `.gitignore` directory.
*   The `dataset` directory stores the face embeddings database.
*   `embeddings.py` extracts face embeddings from images using deep learning techniques.
*   `recognizewithcam.py` uses a webcam feed to recognize faces in real-time.
*   `requirements.txt` outlines the project dependencies.

**Features**
------------

*   Create a database of face embeddings:
    *   Face detection using pre-trained models
    *   Face embedding extraction using deep learning techniques
    *   Database storage using pickling
*   Recognize faces in real-time using a webcam feed:
    *   Real-time face detection
    *   Face comparison with stored embeddings
    *   Output of recognized faces

**Technologies Used**
---------------------

*   **Python 3.9+**: Primary programming language
*   **OpenCV 4.5+**: Computer vision library
*   **Dlib 19.22+**: Deep learning library
*   **Pickle 4.0+**: Data serialization library

**How to Run the Project**
-------------------------

### 1. Install Dependencies:

    pip install -r requirements.txt

### 2. Create Face Embeddings:

    python embeddings.py --input-path /path/to/dataset

### 3. Recognize Faces in Real-Time:

    python recognizewithcam.py

### Running with Camera

Ensure you have a webcam configured on your system.

Run `recognizewithcam.py` to start the face recognition service.


**Contributing**
---------------

1. Clone the repository.
2. Install project dependencies.
3. Implement new features or fix issues.

Feel free to open an issue or submit a pull request to contribute to the Facial-Recognition-System project.
