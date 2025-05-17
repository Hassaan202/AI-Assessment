from flask import Flask, request, jsonify, render_template, Response
import os
import json
import base64
import io
from PIL import Image
from dotenv import load_dotenv
import google.generativeai as genai

# Create Flask app
app = Flask(__name__)

# Load environment variables
load_dotenv()

# Configure Gemini API
api_key = os.getenv('GOOGLE_API_KEY')
if api_key:
    genai.configure(api_key=api_key)
    genai_model = genai.GenerativeModel('models/gemini-2.0-flash')
else:
    print("Warning: GOOGLE_API_KEY not found in environment variables!")

# Simplified class names for demo purposes
class_names = [
    'Pepper__bell___Bacterial_spot',
    'Potato___healthy',
    'Tomato_Leaf_Mold',
    'Tomato__Tomato_YellowLeaf__Curl_Virus',
    'Tomato_Bacterial_spot',
    'Tomato_Septoria_leaf_spot',
    'Tomato_healthy',
    'Tomato_Spider_mites_Two_spotted_spider_mite',
    'Tomato_Early_blight',
    'Tomato__Target_Spot',
    'Pepper__bell___healthy',
    'Potato___Late_blight',
    'Tomato_Late_blight',
    'Potato___Early_blight',
    'Tomato__Tomato_mosaic_virus'
]

def get_disease_info(disease_name):
    if not api_key:
        return json.dumps({
            "description": "API key not configured",
            "symptoms": "Please check server configuration",
            "treatment": "Please check server configuration"
        })
    
    prompt = f"""Please provide detailed information about the plant disease: {disease_name} relevant to Pakistan. Donnot include the word Pakistan too many times in the response. Also donnot make any text bold or other formatting as I want the response to be in plain text.
    Format the response as a valid JSON object with the following structure:
    {{
        "description": "Brief description of the disease",
        "symptoms": "Common symptoms",
        "treatment": "Recommended treatment methods"
    }}
    Keep each section concise and informative. Ensure the response is a valid JSON object."""
    
    try:
        response = genai_model.generate_content(prompt)
        
        # Try to parse the response as JSON
        try:
            # Clean the response text to ensure it's valid JSON
            response_text = response.text.strip()
            # Remove any markdown code block indicators if present
            response_text = response_text.replace('```json', '').replace('```', '').strip()
            
            parsed_response = json.loads(response_text)
            return json.dumps(parsed_response)  # Return formatted JSON string
        except json.JSONDecodeError:
            # Try to extract JSON-like structure
            try:
                # Find content between curly braces
                import re
                json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
                if json_match:
                    json_str = json_match.group(0)
                    parsed_response = json.loads(json_str)
                    return json.dumps(parsed_response)
            except Exception:
                pass
            
            # If all parsing attempts fail, return a structured error response
            error_response = {
                "description": "Error parsing response",
                "symptoms": "Please try again or consult a plant expert",
                "treatment": "Please try again or consult a plant expert"
            }
            return json.dumps(error_response)
            
    except Exception:
        error_response = {
            "description": "Information not available",
            "symptoms": "Please consult a plant expert",
            "treatment": "Please consult a plant expert"
        }
        return json.dumps(error_response)

def get_mock_prediction(image_bytes):
    """Mock prediction function that returns a placeholder response"""
    try:
        # Just open the image to verify it's valid, but don't use a model
        image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
        
        # Return a simulated prediction with random data
        import random
        
        # Pick a random class for demo
        random_class_index = random.randint(0, len(class_names) - 1)
        primary_class = class_names[random_class_index]
        
        # Create random confidence values
        primary_confidence = random.uniform(70.0, 95.0)
        
        # Create random top predictions
        indices = list(range(len(class_names)))
        random.shuffle(indices)
        top_indices = indices[:5]  # Take 5 random indices
        
        # Ensure our primary prediction is included
        if random_class_index not in top_indices:
            top_indices[0] = random_class_index
            
        # Create confidences for top predictions
        confidences = [random.uniform(20.0, 95.0) for _ in top_indices]
        
        # Ensure primary class has highest confidence
        max_confidence = max(confidences)
        primary_index = top_indices.index(random_class_index)
        confidences[primary_index] = max(max_confidence, primary_confidence)
        
        # Sort by confidence
        predictions = sorted(
            [{'class': class_names[idx], 'confidence': conf} for idx, conf in zip(top_indices, confidences)],
            key=lambda x: x['confidence'],
            reverse=True
        )
        
        return {
            'class': primary_class,
            'confidence': primary_confidence,
            'top_predictions': predictions
        }
        
    except Exception as e:
        return {'error': str(e)}

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'})
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'})
    
    img_bytes = file.read()
    prediction = get_mock_prediction(img_bytes)
    return jsonify(prediction)

@app.route('/api/predict', methods=['POST'])
def api_predict():
    if not request.json or 'image' not in request.json:
        return jsonify({'error': 'No image data'})
    
    try:
        img_data = base64.b64decode(request.json['image'])
        prediction = get_mock_prediction(img_data)
        return jsonify(prediction)
    except Exception as e:
        return jsonify({'error': str(e)})

@app.route('/api/disease-info', methods=['POST'])
def disease_info():
    if not request.json or 'disease' not in request.json:
        return jsonify({'error': 'No disease name provided'})
    
    disease_name = request.json['disease']
    info = get_disease_info(disease_name)
    if isinstance(info, str):
        try:
            info = json.loads(info)
        except Exception:
            info = {"description": "Error parsing response", "symptoms": "", "treatment": ""}
    return jsonify(info)

# Create app instance for Vercel
app = app