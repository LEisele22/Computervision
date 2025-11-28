

import cv2
import numpy as np
import time
import csv
import os
from collections import deque
from datetime import datetime

# ----------------------------
# Anpassbare Pfade / Einstellungen
# ----------------------------
MODELS = {
    "1": "models/yolov5n.onnx",   # Modell 1 (klein)
    "2": "models/yolov5m.onnx",   # Modell 2 (mittel) - passe Pfad an, falls nötig
}

IMG_SIZE = 640            # Eingabegröße für das Modell (häufig 640 für yolov5 export)
CONF_THRESH = 0.25        # minimaler confidence threshold (objectness * class_prob)
NMS_THRESH = 0.45
CAM_ID = 0                # Kamera-ID (0 ist meist die eingebaute Kamera)
OUTPUT_CSV = "detections.csv"

# YOLOv5 benutzt COCO-Classes (80). Hier ein Standard-Array (index = class_id).
# Du kannst alternativ eine 'coco.names' Datei laden, wenn du willst.
COCO_CLASSES = [
 "person","bicycle","car","motorcycle","airplane","bus","train","truck","boat","traffic light",
 "fire hydrant","stop sign","parking meter","bench","bird","cat","dog","horse","sheep","cow",
 "elephant","bear","zebra","giraffe","backpack","umbrella","handbag","tie","suitcase","frisbee",
 "skis","snowboard","sports ball","kite","baseball bat","baseball glove","skateboard","surfboard","tennis racket","bottle",
 "wine glass","cup","fork","knife","spoon","bowl","banana","apple","sandwich","orange",
 "broccoli","carrot","hot dog","pizza","donut","cake","chair","couch","potted plant","bed",
 "dining table","toilet","tv","laptop","mouse","remote","keyboard","cell phone","microwave","oven",
 "toaster","sink","refrigerator","book","clock","vase","scissors","teddy bear","hair drier","toothbrush"
]

# ----------------------------
# Hilfsfunktionen
# ----------------------------
def load_net(onnx_path):
    if not os.path.isfile(onnx_path):
        raise FileNotFoundError(f"ONNX model not found: {onnx_path}")
    print(f"[INFO] Lade ONNX Model: {onnx_path}")
    net = cv2.dnn.readNetFromONNX(onnx_path)
    # CPU-only: stelle sicher, dass OpenCV CPU-Backend verwendet (Standard)
    # Falls du GPU (OpenCV + CUDA) nutzen würdest, könntest du hier setPreferableBackend/Target setzen.
    return net

def preprocess(image, img_size=IMG_SIZE):
    """
    Resize, BGR->RGB, normalize to [0,1], transpose to NCHW and return blob.
    """
    h0, w0 = image.shape[:2]
    # letterbox resize to keep aspect ratio
    r = img_size / max(h0, w0)
    new_unpad = (int(round(w0 * r)), int(round(h0 * r)))
    resized = cv2.resize(image, new_unpad, interpolation=cv2.INTER_LINEAR)
    # pad
    dw = img_size - new_unpad[0]
    dh = img_size - new_unpad[1]
    top, bottom = dh // 2, dh - (dh // 2)
    left, right = dw // 2, dw - (dw // 2)
    padded = cv2.copyMakeBorder(resized, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(114,114,114))
    img = padded[:, :, ::-1]  # BGR->RGB
    img = img.astype(np.float32) / 255.0
    img = np.transpose(img, (2,0,1))  # HWC->CHW
    img = np.expand_dims(img, 0)      # add batch dimension
    return img, r, (left, top), (w0, h0), new_unpad

def postprocess(outputs, img_shape, pad, ratio, conf_thres=CONF_THRESH, nms_thres=NMS_THRESH):
    """
    outputs: output from net.forward() (expected shape (1, N, 85) where N = num boxes)
    returns: boxes in (x1,y1,x2,y2), confidences, class_ids
    """
    # Typical YOLOv5 ONNX output shape: (1, N, 85)
    preds = outputs[0]  # shape (N,85)
    boxes = []
    confidences = []
    class_ids = []

    w0, h0 = img_shape
    pad_x, pad_y = pad
    for det in preds:
        # det: [cx, cy, w, h, obj_conf, class1, class2, ...]
        obj_conf = float(det[4])
        if obj_conf < 1e-6:
            continue
        class_scores = det[5:]
        class_id = int(np.argmax(class_scores))
        cls_conf = float(class_scores[class_id])
        conf = obj_conf * cls_conf
        if conf < conf_thres:
            continue
        # box in format center x,y,w,h (normalized w.r.t. model input IMG_SIZE) - need to map back to original image
        cx, cy, bw, bh = det[0:4]
        # convert from model-space to pixel coords (account for padding and scaling)
        # The preprocess used letterbox with padding (left,top) and scaling ratio r
        # cx,cy,bw,bh are in pixels relative to IMG_SIZE (if export used that), so convert:
        x = (cx - bw / 2.0)
        y = (cy - bh / 2.0)
        # remove pad, divide by ratio to original
        x = (x - pad_x) / ratio
        y = (y - pad_y) / ratio
        bw = bw / ratio
        bh = bh / ratio
        x1 = max(0, int(x))
        y1 = max(0, int(y))
        x2 = min(w0, int(x + bw))
        y2 = min(h0, int(y + bh))
        boxes.append([x1, y1, x2 - x1, y2 - y1])  # use x,y,w,h for NMSBoxes
        confidences.append(float(conf))
        class_ids.append(int(class_id))

    # NMS (uses boxes in x,y,w,h)
    idxs = cv2.dnn.NMSBoxes(boxes, confidences, conf_thres, nms_thres)
    final_boxes = []
    final_scores = []
    final_classids = []
    if len(idxs) > 0:
        for i in idxs.flatten():
            final_boxes.append(boxes[i])
            final_scores.append(confidences[i])
            final_classids.append(class_ids[i])
    return final_boxes, final_scores, final_classids

# ----------------------------
# Haupt-Loop
# ----------------------------
def main():
    # Start camera
    cap = cv2.VideoCapture(CAM_ID, cv2.CAP_DSHOW)  # on Windows, CAP_DSHOW often more reliable
    if not cap.isOpened():
        print("[ERROR] Kann Kamera nicht öffnen. Prüfe CAM_ID.")
        return

    # Start with model "1" (falls vorhanden)
    current_model_key = "1" if "1" in MODELS else list(MODELS.keys())[0]
    net = load_net(MODELS[current_model_key])

    # CSV Logging vorbereiten
    csv_path = OUTPUT_CSV
    csv_file = open(csv_path, mode='w', newline='', encoding='utf-8')
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow(["timestamp", "frame_idx", "class_id", "class_name", "conf", "x", "y", "w", "h", "model"])

    frame_idx = 0
    paused = False
    fps_deque = deque(maxlen=30)
    t_prev = time.time()

    print("[INFO] Drücke 'q' zum Beenden, '1'/'2' zum Modell wechseln, 'p' pause/resume.")

    while True:
        if not paused:
            ret, frame = cap.read()
            if not ret:
                print("[WARN] Frame capture failed, beende.")
                break
        else:
            # wenn pausiert, nur Warte-Tasten abfragen
            time.sleep(0.05)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('p'):
                paused = not paused
                print("[INFO] Paused toggled:", paused)
            if key == ord('q'):
                break
            continue

        t0 = time.time()
        img_in, ratio, pad, orig_shape, new_unpad = preprocess(frame, IMG_SIZE)
        # create blob and forward
        blob = np.ascontiguousarray(img_in)
        net.setInput(blob)
        outs = net.forward()  # expected shape (1,N,85)

        # postprocess
        boxes, scores, classids = postprocess(outs, orig_shape, pad, ratio, CONF_THRESH, NMS_THRESH)

        # draw boxes
        for (box, score, cid) in zip(boxes, scores, classids):
            x, y, w, h = box
            x2 = x + w
            y2 = y + h
            label = f"{COCO_CLASSES[cid] if cid < len(COCO_CLASSES) else cid}: {score:.2f}"
            # bbox
            cv2.rectangle(frame, (x, y), (x2, y2), (0, 255, 0), 2)
            # label
            t_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]
            cv2.rectangle(frame, (x, y - t_size[1] - 6), (x + t_size[0] + 6, y), (0,255,0), -1)
            cv2.putText(frame, label, (x + 3, y - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 1)

            # log detection to CSV
            timestamp = datetime.utcnow().isoformat()
            csv_writer.writerow([timestamp, frame_idx, cid, COCO_CLASSES[cid] if cid < len(COCO_CLASSES) else str(cid),
                                 f"{score:.4f}", x, y, w, h, MODELS[current_model_key]])
            csv_file.flush()

        # FPS
        t1 = time.time()
        dt = t1 - t_prev if frame_idx > 0 else (t1 - t0)
        t_prev = t1
        fps = 1.0 / (t1 - t0) if (t1 - t0) > 0 else 0.0
        fps_deque.append(fps)
        fps_smooth = sum(fps_deque)/len(fps_deque)

        # Anzeige
        cv2.putText(frame, f"Model: {os.path.basename(MODELS[current_model_key])}", (10,20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,255), 2)
        cv2.putText(frame, f"FPS: {fps_smooth:.1f}", (10,50),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,255), 2)
        cv2.imshow("YOLOv5 - Live", frame)

        frame_idx += 1

        # Tastatureingaben
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('p'):
            paused = not paused
        elif key == ord('1') and "1" in MODELS:
            current_model_key = "1"
            net = load_net(MODELS[current_model_key])
        elif key == ord('2') and "2" in MODELS:
            current_model_key = "2"
            net = load_net(MODELS[current_model_key])
        elif key == ord('r'):
            # rotiere CSV
            csv_file.close()
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            csv_path = f"detections_{timestamp}.csv"
            csv_file = open(csv_path, mode='w', newline='', encoding='utf-8')
            csv_writer = csv.writer(csv_file)
            csv_writer.writerow(["timestamp", "frame_idx", "class_id", "class_name", "conf", "x", "y", "w", "h", "model"])
            print(f"[INFO] Neues CSV: {csv_path}")

    # Cleanup
    csv_file.close()
    cap.release()
    cv2.destroyAllWindows()
    print("[INFO] Beendet. Letzte CSV:", csv_path)

if __name__ == "__main__":
    main()
