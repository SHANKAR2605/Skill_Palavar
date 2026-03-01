import cv2
import time

# How many consecutive frames with no eyes trigger alert
EYE_MISSING_FRAMES_THRESHOLD = 20  

# Load Haar cascades for face and eyes
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
eye_cascade  = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_eye.xml")

cap = cv2.VideoCapture(0)  # 0 = default camera

consecutive_no_eye = 0
last_alert_time = 0

print("Starting camera. Press 'q' to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame")
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(80, 80))

    eyes_found_in_frame = False

    for (x, y, w, h) in faces:
        face_roi_gray = gray[y:y+h, x:x+w]
        face_roi_color = frame[y:y+h, x:x+w]

        eyes = eye_cascade.detectMultiScale(face_roi_gray, scaleFactor=1.1, minNeighbors=4, minSize=(15, 15))

        if len(eyes) > 0:
            eyes_found_in_frame = True
            for (ex, ey, ew, eh) in eyes:
                cv2.rectangle(face_roi_color, (ex, ey), (ex+ew, ey+eh), (0,255,0), 2)

        cv2.rectangle(frame, (x, y), (x+w, y+h), (255,0,0), 2)
        break  # Only process the first face

    if eyes_found_in_frame:
        consecutive_no_eye = 0
    else:
        consecutive_no_eye += 1

    if consecutive_no_eye >= EYE_MISSING_FRAMES_THRESHOLD:
        alert_text = "ALERT: Face not recognized door locked / Access deined - STOP"
        cv2.putText(frame, alert_text, (30, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,0,255), 3)
        now = time.time()
        if now - last_alert_time > 1.0:
            print(f"[{time.strftime('%H:%M:%S')}] {alert_text}")
            last_alert_time = now
    else:
        cv2.putText(frame, f"Access granted door open ({consecutive_no_eye})", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)

    cv2.imshow("Driver Monitor (press q to quit)", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
print("Camera released, exiting.")