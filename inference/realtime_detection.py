import cv2
from ultralytics import YOLO
from playsound import playsound
from datetime import datetime
import threading
import os

# ---------------- SETTINGS ----------------

MODEL_PATH = "../models/animal_v2/weights/best.pt"
ALARM_SOUND = "../assets/alarm.wav"
LOG_FILE = "../logs/detection_log.txt"

CARNIVORES = {"lion", "tiger"}

os.makedirs("../logs", exist_ok=True)

# ---------------- LOAD MODEL ----------------

model = YOLO(MODEL_PATH)

print("✅ Model Loaded")
print("Classes:", model.names)

# ---------------- SOUND FUNCTION ----------------

def play_alarm():
    playsound(ALARM_SOUND)

# ---------------- START CAMERA ----------------

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("❌ Camera not detected")
    exit()

print("📷 Camera Started")

# ---------------- REALTIME LOOP ----------------

while True:

    ret, frame = cap.read()

    if not ret:
        break

    results = model.predict(frame, conf=0.4, verbose=False)
    result = results[0]

    carnivore_count = 0
    detected_animals = []

    for box in result.boxes:

        cls_id = int(box.cls[0])
        conf = float(box.conf[0])
        label = model.names[cls_id]

        x1, y1, x2, y2 = map(int, box.xyxy[0])

        label_lower = label.lower()

        if label_lower in CARNIVORES:
            color = (0, 0, 255)
            carnivore_count += 1
        else:
            color = (255, 0, 0)

        detected_animals.append((label, conf))

        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

        text = f"{label} {conf:.2f}"

        cv2.putText(frame,
                    text,
                    (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    color,
                    2)

    # ---------------- DISPLAY COUNT ----------------

    cv2.putText(frame,
                f"Carnivores: {carnivore_count}",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 0, 255),
                3)

    # ---------------- ALERT SYSTEM ----------------

    if carnivore_count > 0:

        cv2.putText(frame,
                    "⚠ CARNIVORE ALERT ⚠",
                    (20, 80),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0, 0, 255),
                    3)

        # play alarm in background thread
        threading.Thread(target=play_alarm, daemon=True).start()

        # save snapshot
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"../logs/alert_{timestamp}.jpg"
        cv2.imwrite(filename, frame)

        # log detection
        with open(LOG_FILE, "a") as f:
            for animal, conf in detected_animals:
                f.write(f"{datetime.now()} - {animal} - {conf:.2f}\n")

    # ---------------- SHOW WINDOW ----------------

    cv2.imshow("Wild Animal Monitoring System", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()