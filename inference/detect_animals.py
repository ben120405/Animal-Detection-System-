import cv2
import os
from ultralytics import YOLO

# -------------------- SETTINGS --------------------

MODEL_PATH = "../models/best.pt"
TEST_FOLDER = "../dataset/test/images"
CARNIVORES = ["lion", "tiger"]

# -------------------- LOAD MODEL --------------------

model = YOLO(MODEL_PATH)
print("✅ Model Loaded")

print("Model class names:", model.names)

# -------------------- LOOP THROUGH ALL IMAGES --------------------

for image_name in os.listdir(TEST_FOLDER):

    image_path = os.path.join(TEST_FOLDER, image_name)

    if not image_path.lower().endswith((".jpg", ".jpeg", ".png")):
        continue

    print(f"\n🔍 Processing: {image_name}")

    results = model.predict(image_path, conf=0.4)
    result = results[0]

    image = cv2.imread(image_path)

    carnivore_count = 0

    for box in result.boxes:
        cls_id = int(box.cls[0])
        conf = float(box.conf[0])
        label = model.names[cls_id]

        x1, y1, x2, y2 = map(int, box.xyxy[0])

        if label.lower() in CARNIVORES:
            color = (0, 0, 255)  # RED
            carnivore_count += 1
        else:
            color = (255, 0, 0)  # BLUE

        cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)

        text = f"{label} {conf:.2f}"
        cv2.putText(image, text, (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    print(f"🐾 Carnivorous Animals Detected: {carnivore_count}")

    # Save output inside inference folder
    output_path = f"output_{image_name}"
    cv2.imwrite(output_path, image)

print("\n✅ All images processed.")
