import os
import numpy as np
import tensorflow as tf
from flask import Flask, request, jsonify, render_template
from werkzeug.utils import secure_filename
from PIL import Image

app = Flask(__name__)


MODEL_PATH = "Xception_Featured.h5"  
try:
    model = tf.keras.models.load_model(MODEL_PATH)
    print("Model loaded successfully!")
except Exception as e:
    print("Error loading model:", e)


ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}


UPLOAD_FOLDER = "uploads"
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def preprocess_image(image_path):
    img = Image.open(image_path).convert("RGB")
    img = img.resize((299, 299)) 
    img_array = np.array(img) / 255.0 
    img_array = np.expand_dims(img_array, axis=0)  
    return img_array

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files['file']

    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400

    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        file_path = os.path.join(UPLOAD_FOLDER, filename)
        
        
        file.save(file_path)

        
        img_array = preprocess_image(file_path)
        prediction = model.predict(img_array)[0][0]  

    
        result = "Deepfake" if prediction > 0.5 else "Real"
        confidence = round(prediction * 100, 2)

        return jsonify({"result": result, "confidence": f"{confidence}%"}), 200

    return jsonify({"error": "Invalid file type"}), 400

if __name__ == '__main__':
    app.run(debug=True)
