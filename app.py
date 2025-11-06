import os
import io
import base64
import cv2
import numpy as np
from flask import Flask, render_template, request, jsonify
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array

# Initialize Flask app
app = Flask(__name__, template_folder='app/static')

# Load the trained model
model = load_model('models/mobilenet_emotion.h5')

# Emotion labels (must match your dataset)
emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']

# Home route
@app.route('/')
def index():
    return render_template('index.html')

# Prediction route
@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()

    # Extract base64 image from JSON
    image_data = data['image']
    image_data = image_data.split(',')[1]
    image_bytes = base64.b64decode(image_data)
    img_array = np.frombuffer(image_bytes, np.uint8)
    frame = cv2.imdecode(img_array, cv2.IMREAD_COLOR)

    # Convert to RGB (since model expects 3 channels)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    face_detector = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    faces = face_detector.detectMultiScale(gray, 1.3, 5)

    if len(faces) == 0:
        return jsonify({'label': 'No face detected', 'prob': 0.0})

    # Take the first detected face
    (x, y, w, h) = faces[0]
    face_roi = rgb[y:y+h, x:x+w]

    # Resize to match model input (96x96)
    face_resized = cv2.resize(face_roi, (96, 96))
    face_resized = face_resized.astype("float") / 255.0
    face_array = img_to_array(face_resized)
    face_array = np.expand_dims(face_array, axis=0)

    # Predict emotion
    preds = model.predict(face_array)[0]
    label = emotion_labels[preds.argmax()]
    prob = float(np.max(preds))

    return jsonify({'label': label, 'prob': prob})

if __name__ == '__main__':
    app.run(debug=True)
