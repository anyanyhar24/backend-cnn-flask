import os
import base64
import numpy as np
from flask import Flask, request, jsonify
from tensorflow import keras
from PIL import Image
import io
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

# load model
model_path = os.path.join('model_native', 'model_cnn.h5')
try:
    model = keras.models.load_model(model_path)
    model_valid = True
except Exception as e:
    model_valid = False
    print(f"Failed to load the model: {str(e)}")
    
# Set threshold probability
threshold = 0.5

@app.route("/")
def main():
    return {
      'message': 'Hello World!',
      'model_valid': model_valid
    }
    
def preprocess_image(image):
    # Resize the image to match model input shape
    image = image.resize((224, 224))
    # Convert the image to numpy array
    image_array = np.array(image)
    # Normalize the image
    image_array = image_array / 255.0
    # Expand dimensions to create batch dimension
    image_array = np.expand_dims(image_array, axis=0)
    return image_array

@app.route("/predict", methods=["POST"])
def predict_image():
    # Check if the model is valid
    if not model_valid:
        return jsonify({"error": "Invalid model"}), 500

    # Check if the request contains an image file
    if "image" not in request.files:
        return jsonify({"error": "No image provided"}), 400

    try:
        image_file = request.files["image"]
        image = Image.open(image_file)
        preprocessed_image = preprocess_image(image)
        predictions = model.predict(preprocessed_image)
        labels = ['apel', 'alpukat', 'pisang', 'belimbing', 'buah naga', 'anggur', 'sirsak', 'jambu batu', 'kiwi', 'mangga', 'manggis', 'jeruk', 'pepaya', 'rambutan', 'semangka']
        
        predicted_labels = [labels[i] for i in np.argmax(predictions, axis=1)]
        
        return jsonify({"predictions": predicted_labels})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=int(os.environ.get("PORT", 5050)))