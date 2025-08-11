# faceMesh Module
A Python project using MediaPipe Face Mesh and OpenCV to detect and track facial landmarks in real time.

Features
- Detects up to maxFaces faces simultaneously.

- Returns precise (x, y) coordinates for each facial landmark.

- Draws mesh connections on the face using MediaPipe’s tessellation.

- Adjustable detection and tracking confidence thresholds.

- Real-time webcam preview with FPS counter.

Requirements

Install the dependencies:

    pip install opencv-python mediapipe
Usage

Run the module directly

    python faceMeshModule.py

This will:

- Open your webcam

- Detect faces

- Draw the mesh overlay

- Print the coordinates of the first detected face

File Structure
    - faceMeshModule.py – Core Face Mesh Detector class
    - faceMeshBasics.py – Basic functionality for us to get running and started
    -faceMeshTesting.py – Testing script for detector.
