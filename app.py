from flask import Flask, request, jsonify
from ultralytics import YOLO
import cv2
import numpy as np

app = Flask(__name__)
model = YOLO('yolov8n.pt')

@app.route('/predict', methods=['POST'])
def predict():
    file = request.files['image'].read()
    npimg = np.frombuffer(file, np.uint8)
    frame = cv2.imdecode(npimg, cv2.IMREAD_COLOR)

    # Rastreamento de pessoas
    results = model.track(frame, persist=True, classes=[0])
    
    # Extrair IDs e coordenadas
    detections = []
    for r in results[0].boxes:
        if r.id is not None:
            detections.append({
                "id": int(r.id),
                "bbox": r.xyxy[0].tolist(),
                "conf": float(r.conf)
            })

    return jsonify({"detections": detections})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
