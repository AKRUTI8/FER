from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
import tensorflow as tf
import numpy as np
import cv2
import pickle
import base64
import os

app = Flask(__name__)
CORS(app)

# Load the model and category mapping
model = tf.keras.models.load_model('my_model.keras')
with open('category_map.pkl', 'rb') as f:
    category_mapping = pickle.load(f)

def preprocess_image(image_data):
    # Decode base64 image
    nparr = np.frombuffer(base64.b64decode(image_data.split(',')[1]), np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    # Resize to match model input
    img = cv2.resize(img, (224, 224))
    
    # Convert to RGB
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # Normalize
    img = img.astype(np.float32) / 255.0
    
    # Add batch dimension
    img = np.expand_dims(img, axis=0)
    
    return img

@app.route('/')
def home():
    return send_file('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get image data from request
        data = request.json
        image_data = data['image']
        
        # Preprocess image
        processed_image = preprocess_image(image_data)
        
        # Make prediction
        predictions = model.predict(processed_image)
        predicted_class = np.argmax(predictions[0])
        confidence = float(predictions[0][predicted_class])
        
        # Get emotion label
        emotion = category_mapping[predicted_class]
        
        return jsonify({
            'emotion': emotion,
            'confidence': confidence
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, port=5000) 