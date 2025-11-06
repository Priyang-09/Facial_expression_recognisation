# app/server.py
import io, base64
import numpy as np
import cv2
from flask import Flask, request, jsonify, send_from_directory
from tensorflow.keras.models import load_model
from PIL import Image
import tensorflow as tf

LABELS = ['Angry','Disgust','Fear','Happy','Sad','Surprise','Neutral']
MODEL_PATH = "../models/mobilenet_emotion.h5"

print("📦 Loading model...")
model = load_model(MODEL_PATH)
print("✅ Model loaded")

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
app = Flask(__name__, static_folder='static')

def decode_base64_image(data_url):
    header, data = data_url.split(',', 1)
    decoded = base64.b64decode(data)
    img = Image.open(io.BytesIO(decoded)).convert('RGB')
    return np.array(img)

def preprocess_face(face_img):
    img = cv2.resize(face_img, (96, 96))
    img = img.astype('float32') / 255.0
    img = np.expand_dims(img, 0)
    return img

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    img_np = decode_base64_image(data['image'])
    gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)

    if len(faces) == 0:
        return jsonify({'label':'No face','prob':0.0})

    x, y, w, h = faces[0]
    face = img_np[y:y+h, x:x+w]
    x_in = preprocess_face(face)
    preds = model.predict(x_in)[0]
    label = LABELS[int(np.argmax(preds))]
    prob = float(np.max(preds))
    return jsonify({'label': label, 'prob': prob})

@app.route('/')
def index():
    return send_from_directory('static', 'index.html')

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000, debug=True)
