import cv2
import mediapipe as mp

webcam = cv2.VideoCapture(0)
face_detection_solution = mp.solutions.face_detection
face_detection = face_detection_solution.FaceDetection()
face_draw = mp.solutions.drawing_utils

while True:
    verify, frame = webcam.read()

    if not verify:
        break
    face_list = face_detection.process(frame)
    face_list_detections = face_list.detections
    if face_list_detections:
        for face in face_list_detections:
            face_draw.draw_detection(frame, face)
    cv2.imshow('Face Detection', frame)
    if cv2.waitKey(5) == 27:
        break
webcam.release()
cv2.destroyAllWindows()
