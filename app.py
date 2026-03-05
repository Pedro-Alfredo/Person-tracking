import os
from flask import Flask, request, jsonify
from ultralytics import YOLO
import cv2
import numpy as np

app = Flask(__name__)

# Carrega o modelo Nano (mais leve de todos)
# O Render vai baixar isso automaticamente na primeira execução
model = YOLO('yolov8n.pt') 

@app.route('/')
def home():
    return "Sistema de Rastreamento Ativo!"

@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return jsonify({"error": "Nenhuma imagem enviada"}), 400
    
    file = request.files['image'].read()
    npimg = np.frombuffer(file, np.uint8)
    frame = cv2.imdecode(npimg, cv2.IMREAD_COLOR)

    # Executa a detecção apenas de pessoas (classe 0)
    # imgsz=320 reduz a resolução internamente para economizar RAM no Render
    results = model.predict(frame, conf=0.25, classes=[0], imgsz=320)
    
    detections = []
    for r in results[0].boxes:
        detections.append({
            "bbox": r.xyxy[0].tolist(), # Coordenadas [x1, y1, x2, y2]
            "conf": round(float(r.conf), 2)
        })

    return jsonify({
        "status": "sucesso",
        "quantidade_pessoas": len(detections),
        "detections": detections
    })

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port)
