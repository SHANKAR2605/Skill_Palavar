import cv2
import os

name = "user1"  # change to the name of the person
folder = f"authorized_faces/{name}"
os.makedirs(folder, exist_ok=True)

cap = cv2.VideoCapture(0)
count = 0
while count < 20:  # capture 20 images
    ret, frame = cap.read()
    if not ret:
        break
    cv2.imshow("Capture Face", frame)
    cv2.imwrite(f"{folder}/{count}.jpg", frame)
    count += 1
    if cv2.waitKey(100) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
print("Face images captured")