import cv2
import dlib
import numpy as np


face_detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")


sunglasses = cv2.imread('sunglasses2.png', cv2.IMREAD_UNCHANGED)  


def get_landmarks(image, face_rect):
    landmarks = predictor(image, face_rect)
    return landmarks


def overlay_sunglasses(frame, landmarks, sunglasses, scaling_factor=0.9):
    
    left_eye = (landmarks.part(36).x, landmarks.part(36).y)
    right_eye = (landmarks.part(45).x, landmarks.part(45).y)

    
    eye_dist = np.linalg.norm(np.array(right_eye) - np.array(left_eye))
    
    
    sunglasses_width = int(eye_dist * 2 * scaling_factor)  
    sunglasses_height = int(sunglasses_width * sunglasses.shape[0] / sunglasses.shape[1])

    
    resized_sunglasses = cv2.resize(sunglasses, (sunglasses_width, sunglasses_height))

    
    center = (int((left_eye[0] + right_eye[0]) / 2), int((left_eye[1] + right_eye[1]) / 2))

    
    top_left = (center[0] - sunglasses_width // 2, center[1] - sunglasses_height // 2)
    bottom_right = (top_left[0] + sunglasses_width, top_left[1] + sunglasses_height)

    
    sunglasses_area = resized_sunglasses[:, :, :3]  
    sunglasses_mask = resized_sunglasses[:, :, 3]  

    
    roi = frame[top_left[1]:bottom_right[1], top_left[0]:bottom_right[0]]

    
    if roi.shape[:2] != sunglasses_area.shape[:2]:
        roi = cv2.resize(roi, (sunglasses_area.shape[1], sunglasses_area.shape[0]))

    
    sunglasses_mask = cv2.cvtColor(sunglasses_mask, cv2.COLOR_GRAY2BGR)  
    sunglasses_mask = sunglasses_mask / 255.0  

    
    for c in range(0, 3):
        roi[:, :, c] = roi[:, :, c] * (1 - sunglasses_mask[:, :, c]) + sunglasses_area[:, :, c] * sunglasses_mask[:, :, c]

    
    frame[top_left[1]:bottom_right[1], top_left[0]:bottom_right[0]] = roi

    return frame


cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    
    faces = face_detector(gray)
    
    for face in faces:
        
        landmarks = get_landmarks(gray, face)

        
        frame = overlay_sunglasses(frame, landmarks, sunglasses, scaling_factor=0.8)  

    
    cv2.imshow("Face with Sunglasses", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
