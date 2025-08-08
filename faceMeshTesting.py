import cv2
import mediapipe as mp
import time

cap = cv2.VideoCapture(0)
pTime = 0

mpDraw = mp.solutions.drawing_utils
mpFaceMesh = mp.solutions.face_mesh


from mediapipe.python.solutions.face_mesh_connections import (
    FACEMESH_TESSELATION,  # Full mesh
    FACEMESH_CONTOURS,     # Face outline
    FACEMESH_IRISES        # Eyes
)

# ✅ Initialize FaceMesh
faceMesh = mpFaceMesh.FaceMesh(
    max_num_faces=2,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

while True:
    success, img = cap.read()
    if not success:
        print("❌ Could not read from webcam")
        break

    imgRgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = faceMesh.process(imgRgb)

    if results.multi_face_landmarks:
        for faceLms in results.multi_face_landmarks:

            mpDraw.draw_landmarks(
                image=img,
                landmark_list=faceLms,
                connections=FACEMESH_TESSELATION,
                landmark_drawing_spec=None,
                connection_drawing_spec=mpDraw.DrawingSpec(color=(0, 255, 0), thickness=1, circle_radius=1)
            )


            mpDraw.draw_landmarks(
                image=img,
                landmark_list=faceLms,
                connections=FACEMESH_CONTOURS,
                landmark_drawing_spec=None,
                connection_drawing_spec=mpDraw.DrawingSpec(color=(255, 0, 0), thickness=2)
            )

            # ✅ Optional: draw eyes (irises)
            mpDraw.draw_landmarks(
                image=img,
                landmark_list=faceLms,
                connections=FACEMESH_IRISES,
                landmark_drawing_spec=None,
                connection_drawing_spec=mpDraw.DrawingSpec(color=(0, 0, 255), thickness=2)
            )

    cTime = time.time()
    fps = 1/(cTime-pTime) if pTime else 0
    pTime = cTime

    cv2.putText(img, f'FPS:{int(fps)}', (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 0, 255), 3)
    cv2.imshow("Face Mesh", img)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
