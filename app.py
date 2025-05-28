from flask import Flask, request, jsonify, render_template
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
from PIL import Image
import numpy as np
import io
import os
import requests
import google.generativeai as genai
import pandas as pd

app = Flask(__name__)

# Load the model
MODEL_PATH = 'plant_disease_model_combined.h5'
try:
    model = load_model(MODEL_PATH)
    print("Model loaded successfully")
    print("Model output shape:", model.output_shape)
    # Get number of classes from model's output shape
    num_classes = model.output_shape[-1]
    print("Number of classes:", num_classes)
except Exception as e:
    print("Error loading model:", e)

# Define image size - adjust these values according to your model's requirements
IMG_WIDTH = 224
IMG_HEIGHT = 224

# Define class names based on your actual model's classes
class_names = [
    'Apple___Apple_scab',
    'Apple___Black_rot',
    'Apple___Cedar_apple_rust',
    'Apple___healthy',
    'Background_without_leaves',
    'Blueberry___healthy',
    'Cherry___Powdery_mildew',
    'Cherry___healthy',
    'Corn___Cercospora_leaf_spot Gray_leaf_spot',
    'Corn___Common_rust',
    'Corn___Northern_Leaf_Blight',
    'Corn___healthy',
    'Grape___Black_rot',
    'Grape___Esca_(Black_Measles)',
    'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)',
    'Grape___healthy',
    'Orange___Haunglongbing_(Citrus_greening)',
    'Peach___Bacterial_spot',
    'Peach___healthy',
    'Pepper,_bell___Bacterial_spot',
    'Pepper,_bell___healthy',
    'Potato___Early_blight',
    'Potato___Late_blight',
    'Potato___healthy',
    'Raspberry___healthy',
    'Soybean___healthy',
    'Squash___Powdery_mildew',
    'Strawberry___Leaf_scorch',
    'Strawberry___healthy',
    'Tomato___Bacterial_spot',
    'Tomato___Early_blight',
    'Tomato___Late_blight',
    'Tomato___Leaf_Mold',
    'Tomato___Septoria_leaf_spot',
    'Tomato___Spider_mites Two-spotted_spider_mite',
    'Tomato___Target_Spot',
    'Tomato___Tomato_Yellow_Leaf_Curl_Virus',
    'Tomato___Tomato_mosaic_virus',
    'Tomato___healthy'
]  # These are the standard PlantVillage dataset classes

# Configure Gemini with your API key
GEMINI_API_KEY = 'AIzaSyD4Z-hAlIUsrbXEsZcEfy7iNLp6Q39C8s4'  # Replace with your actual Gemini API key

genai.configure(api_key=GEMINI_API_KEY)

def preprocess_image(image):
    # Convert the image to RGB if it isn't already
    if image.mode != "RGB":
        image = image.convert("RGB")
    
    # Resize the image
    image = image.resize((IMG_WIDTH, IMG_HEIGHT))
    
    # Convert to array and normalize
    image_array = img_to_array(image)
    image_array = image_array / 255.0  # Normalize to [0,1]
    
    # Add batch dimension
    image_array = np.expand_dims(image_array, axis=0)
    
    return image_array

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file uploaded'})

        file = request.files['file']
        image_bytes = file.read()
        image = Image.open(io.BytesIO(image_bytes))
        
        # Preprocess the image
        processed_image = preprocess_image(image)
        
        # Make prediction
        predictions = model.predict(processed_image)
        predicted_class = np.argmax(predictions[0])
        
        # Verify predicted_class is within bounds
        if predicted_class >= len(class_names):
            raise ValueError(f"Predicted class index {predicted_class} is out of range for {len(class_names)} classes")
            
        confidence = float(predictions[0][predicted_class]) * 100  # Convert to percentage

        return jsonify({
            'class_name': class_names[predicted_class],
            'confidence': f'{confidence:.2f}%'  # Format to 2 decimal places
        })

    except Exception as e:
        print("Prediction error:", str(e))  # Add server-side logging
        return jsonify({'error': str(e)})

def get_disease_insight(disease_name):
    model = genai.GenerativeModel("gemini-2.0-flash")
    prompt = (
        f"""
        For the plant disease '{disease_name}', provide ONLY the most important treatment recommendations as short, clear bullet points. 
        Do NOT include explanations, symptoms, or any template text. 
        Only return 4-6 concise treatment bullet points. 
        Example format:\n- Remove infected leaves\n- Apply fungicide\n- Improve air circulation\n"""
    )
    try:
        response = model.generate_content(prompt)
        # Extract only lines that look like bullet points
        lines = response.text.strip().split('\n')
        bullets = [line.strip() for line in lines if line.strip().startswith(('-', '•', '*'))]
        if not bullets:
            # fallback: return the whole response as one point
            bullets = [response.text.strip()]
        return bullets
    except Exception as e:
        return [f"No treatment recommendations available: {str(e)}"]

@app.route('/disease_insight', methods=['POST'])
def disease_insight():
    data = request.json
    disease_name = data.get('disease_name')
    if not disease_name:
        return jsonify({'error': 'No disease name provided'}), 400

    bullets = get_disease_insight(disease_name)
    return jsonify({'insight': bullets}), 200

@app.route('/supplements', methods=['POST'])
def supplements():
    data = request.json
    disease_name = data.get('disease_name')
    df = pd.read_csv('static/supplement_info.csv')
    # Filter for the disease and ignore healthy classes
    filtered = df[(df['disease_name'] == disease_name) & (~df['disease_name'].str.contains('healthy', case=False))]
    supplements = []
    for _, row in filtered.iterrows():
        if pd.notna(row['supplement name']) and pd.notna(row['buy link']):
            supplements.append({
                'name': row['supplement name'],
                'image_url': row['supplement image'],
                'buy_link': row['buy link']
            })
    return jsonify({'supplements': supplements})

if __name__ == '__main__':
    app.run(debug=True)

