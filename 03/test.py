import onnxruntime as ort
import numpy as np
import cv2

# Lade ONNX
net = ort.InferenceSession("models/best.onnx")

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    img = cv2.resize(frame, (640,640))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.transpose(2,0,1)[np.newaxis,...].astype(np.float32)/255.0

    pred = net.run(None, {"images": img})[0]  # [batch, num_boxes, nc+5]
    for det in pred[0]:  # batch=1
        x, y, w, h, conf = det[:5]
        class_probs = det[5:]
        cls = np.argmax(class_probs)
        conf *= class_probs[cls]  # kombiniere obj_conf * class_conf

        if conf > 0.25:
            x1 = int((x - w/2) * frame.shape[1])
            y1 = int((y - h/2) * frame.shape[0])
            x2 = int((x + w/2) * frame.shape[1])
            y2 = int((y + h/2) * frame.shape[0])
            cv2.rectangle(frame, (x1,y1), (x2,y2), (0,255,0), 2)
            cv2.putText(frame, f"{cls} {conf:.2f}", (x1, y1-5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1)

    cv2.imshow("ONNX YOLOv5 Webcam", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
