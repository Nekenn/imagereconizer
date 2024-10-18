from flask import Flask, request, jsonify
import torch
import cv2
import numpy as np
from ultralytics import YOLO

app = Flask(__name__)

# Load YOLOv8 model
model = YOLO('yolov8n.pt')  # You can replace with yolov8s.pt, yolov8m.pt, etc.

@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return jsonify({'error': 'No image uploaded'}), 400

    # Read the uploaded image
    file = request.files['image']
    img_bytes = np.frombuffer(file.read(), np.uint8)
    img = cv2.imdecode(img_bytes, cv2.IMREAD_COLOR)

    # Run YOLOv8 inference
    results = model(img)

    # Parse results
    detections = []
    for box in results[0].boxes:  # Iterate over detected objects
        x1, y1, x2, y2 = box.xyxy[0].tolist()  # Bounding box coordinates
        label = model.names[int(box.cls)]  # Class label
        confidence = box.conf.item()  # Confidence score
        detections.append({
            'label': label,
            'confidence': confidence,
            'box': [x1, y1, x2, y2]
        })

    return jsonify({'detections': detections})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
