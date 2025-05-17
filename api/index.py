from flask import Flask, request, jsonify, render_template, Response
import os
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
import base64
import io
import google.generativeai as genai
import json
from dotenv import load_dotenv
import sys

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

# Define the CNN architecture
class PlantDiseaseCNN(nn.Module):
    def __init__(self, num_classes):
        super(PlantDiseaseCNN, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2)
        )
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(128 * 16 * 16, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

# Class labels
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

# Load model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = PlantDiseaseCNN(len(class_names))

# For Vercel deployment, adjust the model path
model_path = os.path.join(os.path.dirname(__file__), '../plant_model_improved.pth')
try:
    model.load_state_dict(torch.load(model_path, map_location=device))
    print(f"Model loaded successfully from {model_path}")
except Exception as e:
    print(f"Error loading model: {str(e)}")
    # Don't exit as this would stop the serverless function

model.eval()

# Image transformation
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

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

def get_prediction(image_bytes):
    try:
        image = Image.open(io.BytesIO(image_bytes)).convert('RGB')  # Ensure RGB format
        image = transform(image).unsqueeze(0).to(device)
        
        with torch.no_grad():
            outputs = model(image)
            _, predicted = torch.max(outputs, 1)
            probs = torch.nn.functional.softmax(outputs, dim=1)[0]
            
        return {
            'class': class_names[predicted.item()],
            'confidence': probs[predicted].item() * 100,
            'top_predictions': [
                {
                    'class': class_names[i],
                    'confidence': probs[i].item() * 100
                }
                for i in torch.topk(probs, min(5, len(class_names))).indices.tolist()
            ]
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
    prediction = get_prediction(img_bytes)
    return jsonify(prediction)

@app.route('/api/predict', methods=['POST'])
def api_predict():
    if not request.json or 'image' not in request.json:
        return jsonify({'error': 'No image data'})
    
    try:
        img_data = base64.b64decode(request.json['image'])
        prediction = get_prediction(img_data)
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