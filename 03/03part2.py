import cv2
import numpy as np
import time
import csv
import os
from collections import deque
from datetime import datetime
import onnxruntime as ort

# ----------------------------
# Einstellungen
# ----------------------------
MODEL_PATH = "models/best.onnx"  # Dein trainiertes Modell
IMG_SIZE = 640
CONF_THRESH = 0.25
NMS_THRESH = 0.45
CAM_ID = 0
OUTPUT_CSV = "detections.csv"

# Angepasste Klassenliste (dein fine-tuned Modell)
CLASSES = ["Ambulance", "Bus", "Car", "Motorcycle", "Truck", "Other"]  # passe Other ggf. an

# ----------------------------
# Hilfsfunktionen
# ----------------------------
def preprocess(image, img_size=IMG_SIZE):
    h0, w0 = image.shape[:2]
    r = img_size / max(h0, w0)
    new_unpad = (int(round(w0 * r)), int(round(h0 * r)))
    resized = cv2.resize(image, new_unpad, interpolation=cv2.INTER_LINEAR)
    dw = img_size - new_unpad[0]
    dh = img_size - new_unpad[1]
    top, bottom = dh // 2, dh - (dh // 2)
    left, right = dw // 2, dw - (dw // 2)
    padded = cv2.copyMakeBorder(resized, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(114,114,114))
    img = padded[:, :, ::-1].astype(np.float32) / 255.0  # BGR->RGB
    img = np.transpose(img, (2,0,1))
    img = np.expand_dims(img, 0)
    return img, r, (left, top), (w0, h0)

def postprocess(pred, orig_shape, pad, ratio, conf_thres=CONF_THRESH, nms_thres=NMS_THRESH):
    boxes, scores, classids = [], [], []
    w0, h0 = orig_shape
    pad_x, pad_y = pad
    for det in pred[0]:  # batch size = 1
        obj_conf = float(det[4])
        if obj_conf < 1e-6:
            continue
        class_scores = det[5:]
        class_id = int(np.argmax(class_scores))
        cls_conf = float(class_scores[class_id])
        conf = obj_conf * cls_conf
        if conf < conf_thres:
            continue
        cx, cy, bw, bh = det[:4]
        x = (cx - bw / 2 - pad_x) / ratio
        y = (cy - bh / 2 - pad_y) / ratio
        bw /= ratio
        bh /= ratio
        x1 = max(0, int(x))
        y1 = max(0, int(y))
        x2 = min(w0, int(x + bw))
        y2 = min(h0, int(y + bh))
        boxes.append([x1, y1, x2 - x1, y2 - y1])
        scores.append(conf)
        classids.append(class_id)
    idxs = cv2.dnn.NMSBoxes(boxes, scores, conf_thres, nms_thres)
    final_boxes, final_scores, final_classids = [], [], []
    if len(idxs) > 0:
        for i in idxs.flatten():
            final_boxes.append(boxes[i])
            final_scores.append(scores[i])
            final_classids.append(classids[i])
    return final_boxes, final_scores, final_classids

# ----------------------------
# Main
# ----------------------------
def main():
    if not os.path.isfile(MODEL_PATH):
        print(f"[ERROR] Modell nicht gefunden: {MODEL_PATH}")
        return

    print("[INFO] Lade ONNX Modell...")
    session = ort.InferenceSession(MODEL_PATH)

    cap = cv2.VideoCapture(CAM_ID, cv2.CAP_DSHOW)
    if not cap.isOpened():
        print("[ERROR] Kamera nicht verfügbar!")
        return

    csv_file = open(OUTPUT_CSV, mode='w', newline='', encoding='utf-8')
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow(["timestamp", "frame_idx", "class_id", "class_name", "conf", "x", "y", "w", "h"])

    frame_idx = 0
    fps_deque = deque(maxlen=30)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        t0 = time.time()
        img, ratio, pad, orig_shape = preprocess(frame)
        pred = session.run(None, {"images": img.astype(np.float32)})[0]

        boxes, scores, classids = postprocess(pred, orig_shape, pad, ratio)

        # Draw boxes
        for (box, score, cid) in zip(boxes, scores, classids):
            x, y, w, h = box
            x2, y2 = x + w, y + h
            label = f"{CLASSES[cid] if cid < len(CLASSES) else cid}: {score:.2f}"
            cv2.rectangle(frame, (x,y), (x2,y2), (0,255,0), 2)
            t_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]
            cv2.rectangle(frame, (x, y - t_size[1] - 6), (x + t_size[0] + 6, y), (0,255,0), -1)
            cv2.putText(frame, label, (x + 3, y - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 1)

            timestamp = datetime.utcnow().isoformat()
            csv_writer.writerow([timestamp, frame_idx, cid, CLASSES[cid] if cid < len(CLASSES) else str(cid),
                                 f"{score:.4f}", x, y, w, h])
            csv_file.flush()

        fps = 1.0 / (time.time() - t0)
        fps_deque.append(fps)
        fps_smooth = sum(fps_deque)/len(fps_deque)
        cv2.putText(frame, f"FPS: {fps_smooth:.1f}", (10,30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,255), 2)

        cv2.imshow("YOLOv5 Live Detection", frame)
        frame_idx += 1

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break

    csv_file.close()
    cap.release()
    cv2.destroyAllWindows()
    print("[INFO] Beendet. CSV gespeichert:", OUTPUT_CSV)

if __name__ == "__main__":
    main()
