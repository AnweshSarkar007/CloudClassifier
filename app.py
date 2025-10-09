import tensorflow as tf
import numpy as np
from flask import Flask, request, jsonify
from flask_cors import CORS
from PIL import Image
import base64
import io

# Initialize the Flask application
app = Flask(__name__)
CORS(app) # This is important to allow your frontend to communicate with the backend

# --- MODEL LOADING ---
# Load your trained model

MODEL_PATH = r'D:/Projects/Cloud/archive (1)/Two.keras'
model = tf.keras.models.load_model(MODEL_PATH)
model = tf.keras.models.load_model(MODEL_PATH)
CLASS_NAMES = ['Cirriform', 'Clear Sky', 'Cumuliform', 'Cumulonimbus', 'Cumulus', 'Stratiform', 'Stratocumulus'] # Make sure this order is correct!

# --- IMAGE PREPROCESSING ---
def preprocess_image(image_data):
    # The input is a base64-encoded string. We need to decode it.
    image_bytes = base64.b64decode(image_data.split(',')[1])
    
    # Convert bytes to a PIL Image
    img = Image.open(io.BytesIO(image_bytes))

    # Ensure image is RGB (some webcams might send RGBA)
    if img.mode != 'RGB':
        img = img.convert('RGB')
        
    # Resize the image to the size your model expects (224x224)
    img = img.resize((224, 224))
    
    # Convert the image to a NumPy array and expand dimensions for the model
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0) # Create a batch
    
    return img_array

# --- API ENDPOINT ---
@app.route('/predict', methods=['POST'])

# --- This is the NEW, corrected code for Flask ---
# --- This is the NEW, corrected code for Flask ---
def predict():
    # Define your confidence threshold (e.g., 60%)
    CONFIDENCE_THRESHOLD = 0.60 

    data = request.get_json()
    processed_image = preprocess_image(data['image'])
    
    logits = model.predict(processed_image)
    probabilities = tf.nn.softmax(logits).numpy()
    
    # Get the highest confidence score
    max_confidence = float(np.max(probabilities))
    
    # --- Check if the confidence is above the threshold ---
    if max_confidence > CONFIDENCE_THRESHOLD:
        # If it is, proceed as normal
        predicted_index = np.argmax(probabilities)
        predicted_class = CLASS_NAMES[predicted_index]
        confidence_percent = f"{max_confidence*100:.2f}%"
    else:
        # If not, return the "error" message
        predicted_class = "No Cloud Detected"
        confidence_percent = "N/A"
        
    return jsonify({
        'prediction': predicted_class,
        'confidence': confidence_percent
    })
# --- START THE APP ---
if __name__ == '__main__':
    # Use 0.0.0.0 to make it accessible on your local network
    app.run(host='0.0.0.0', port=5000)