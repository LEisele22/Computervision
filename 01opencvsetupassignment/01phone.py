import cv2
import time

# 0 = erste Webcam, 1 = zweite Webcam usw.
# Prüfe mit cap = cv2.VideoCapture(0..n), welche dein Handy ist
cap = cv2.VideoCapture(2)  # DirectShow statt Standard

if not cap.isOpened():
    print("Konnte Kamera nicht öffnen.")
    exit()

while True:
    start = time.time()
    ret, frame = cap.read()
    ret, frame = cap.read()
    if not ret:
        print("Frame konnte nicht gelesen werden.")
        break

    fps = 1 / (time.time() - start)
    cv2.putText(frame, f"FPS: {fps:.2f}", (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    cv2.imshow("DroidCam Webcam", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
