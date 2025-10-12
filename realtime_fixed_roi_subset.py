# realtime_fixed_roi_subset.py
import cv2, numpy as np, tensorflow as tf, json
from collections import deque
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

MODEL = "models/asl_subset_mobilenet.h5"
IMG_SIZE = (160,160)
model = tf.keras.models.load_model(MODEL)
with open("models/class_indices.json") as f:
    class_indices = json.load(f)
inv = {v:k for k,v in class_indices.items()}

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise RuntimeError("Cannot open webcam")

# Fixed ROI coordinates (adjust to put rectangle on right side)
h_offset = 100
w_offset = 350
roi_w = 300
roi_h = 300

smooth = deque(maxlen=10)

print("Place your hand inside the rectangle. Press q to quit.")
while True:
    ret, frame = cap.read()
    if not ret: break
    frame = cv2.flip(frame, 1)
    h, w = frame.shape[:2]

    x1 = w - w_offset
    y1 = h_offset
    x2 = x1 + roi_w
    y2 = y1 + roi_h

    cv2.rectangle(frame, (x1,y1), (x2,y2), (255,0,0), 2)
    roi = frame[y1:y2, x1:x2]
    if roi.size != 0:
        img = cv2.resize(roi, IMG_SIZE)
        inp = preprocess_input(img.astype("float32"))
        inp = np.expand_dims(inp, axis=0)
        pred = model.predict(inp)[0]
        idx = int(pred.argmax())
        smooth.append(idx)
    if smooth:
        smoothed = max(set(smooth), key=smooth.count)
        label = inv[smoothed]
        conf = 100 * float(pred[smoothed])
        cv2.putText(frame, f"{label} {conf:.1f}%", (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,255,0), 2)

    cv2.imshow("Fixed ROI ASL (subset)", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
