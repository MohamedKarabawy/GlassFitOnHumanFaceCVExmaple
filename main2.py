import cv2
import dlib
import numpy as np


face_detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")


sunglasses_image = cv2.imread('sunglasses.png', cv2.IMREAD_UNCHANGED)  


def get_landmarks(image, face_rect):
    landmarks = predictor(image, face_rect)
    return landmarks


def overlay_sunglasses(frame, landmarks, sunglasses, sunglasses_width, sunglasses_height):
    
    left_eye = (landmarks.part(36).x, landmarks.part(36).y)
    right_eye = (landmarks.part(45).x, landmarks.part(45).y)

    
    eye_dist = np.linalg.norm(np.array(right_eye) - np.array(left_eye))

    
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


def capture_image_with_accuracy(sunglasses_width, sunglasses_height):
    
    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to capture image")
            break

        
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        
        faces = face_detector(gray)

        for face in faces:
            
            landmarks = get_landmarks(gray, face)

            
            left_eye = (landmarks.part(36).x, landmarks.part(36).y)
            right_eye = (landmarks.part(45).x, landmarks.part(45).y)
            eye_dist = np.linalg.norm(np.array(right_eye) - np.array(left_eye))

            
            scaling_factor = eye_dist / sunglasses_width  
            sunglasses_scaled_width = int(eye_dist * scaling_factor)
            sunglasses_scaled_height = int(sunglasses_scaled_width * sunglasses_height / sunglasses_width)

            
            frame_with_sunglasses = overlay_sunglasses(frame, landmarks, sunglasses_image, sunglasses_scaled_width, sunglasses_scaled_height)

            
            cv2.putText(frame_with_sunglasses, f"Adjust your head for best fit", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.imshow("Webcam Feed", frame_with_sunglasses)

        
        if cv2.waitKey(1) & 0xFF == ord('c'):
            
            cv2.imwrite("captured_image_with_sunglasses.jpg", frame_with_sunglasses)
            print("Image saved!")
            break

        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    
    cap.release()
    cv2.destroyAllWindows()


sunglasses_width = int(input("Enter the width of the sunglasses (in pixels): "))
sunglasses_height = int(input("Enter the height of the sunglasses (in pixels): "))


capture_image_with_accuracy(sunglasses_width, sunglasses_height)
