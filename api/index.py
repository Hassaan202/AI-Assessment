from flask import Flask, request, jsonify, Response
import os
import json
import base64
import io
from PIL import Image
import random

# Create Flask app
app = Flask(__name__)

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
    # Simplified mock responses for each disease type
    disease_info = {
        "description": f"This is a common disease affecting {disease_name.split('_')[0]} plants.",
        "symptoms": f"Typical symptoms include spots on leaves, wilting, and discoloration.",
        "treatment": "Apply appropriate fungicide or bactericide as recommended by agricultural experts."
    }
    return json.dumps(disease_info)

def get_mock_prediction(image_bytes):
    """Mock prediction function that returns a placeholder response"""
    try:
        # Just open the image to verify it's valid
        image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
        
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

# Modified routes for serverless setup
@app.route('/', defaults={'path': ''})
@app.route('/<path:path>')
def catch_all(path):
    return get_index()

def get_index():
    html_content = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Plant Disease Detection (Demo)</title>
    <link href="https://cdnjs.cloudflare.com/ajax/libs/tailwindcss/2.2.19/tailwind.min.css" rel="stylesheet">
    <style>
        .loading {
            display: none;
        }
        .results {
            display: none;
        }
    </style>
</head>
<body class="bg-gray-100 min-h-screen">
    <div class="container mx-auto px-4 py-8">
        <header class="text-center mb-10">
            <h1 class="text-3xl font-bold text-green-700">Plant Disease Detection</h1>
            <p class="text-gray-600 mt-2">Upload an image of a plant leaf to detect possible diseases</p>
            <div class="mt-2 bg-yellow-100 border-l-4 border-yellow-500 text-yellow-700 p-4 rounded" role="alert">
                <p><strong>Demo Mode:</strong> This is a demonstration version. The predictions are simulated.</p>
            </div>
        </header>
        
        <main class="max-w-2xl mx-auto bg-white rounded-lg shadow-md p-6">
            <form id="upload-form" class="mb-6">
                <div class="mb-4">
                    <label class="block text-gray-700 mb-2" for="image-upload">Upload Image</label>
                    <input type="file" id="image-upload" name="file" accept="image/*" 
                           class="w-full px-3 py-2 border border-gray-300 rounded-md">
                </div>
                <div class="text-center">
                    <button type="submit" class="bg-green-600 text-white px-6 py-2 rounded-md hover:bg-green-700">
                        Analyze Image
                    </button>
                </div>
            </form>
            
            <div id="loading" class="loading text-center py-4">
                <p class="text-gray-700">Analyzing image...</p>
                <div class="mt-2 w-12 h-12 border-4 border-green-600 border-t-transparent rounded-full animate-spin mx-auto"></div>
            </div>
            
            <div id="results" class="results">
                <div class="flex flex-col md:flex-row gap-6">
                    <div class="md:w-1/2">
                        <h3 class="text-lg font-semibold text-gray-800 mb-2">Uploaded Image</h3>
                        <div class="border border-gray-300 rounded-md overflow-hidden">
                            <img id="preview-image" src="#" alt="Uploaded plant image" class="w-full h-auto">
                        </div>
                    </div>
                    
                    <div class="md:w-1/2">
                        <h3 class="text-lg font-semibold text-gray-800 mb-2">Detection Results</h3>
                        <div class="border border-gray-300 rounded-md p-4">
                            <div class="mb-4">
                                <p class="text-gray-700">Detected Disease:</p>
                                <p id="disease-name" class="font-semibold text-green-800 text-xl"></p>
                                <p id="confidence" class="text-sm text-gray-600"></p>
                            </div>
                            
                            <div id="disease-info" class="mt-4 pt-4 border-t border-gray-200">
                                <h4 class="font-semibold text-gray-800 mb-2">Disease Information</h4>
                                <div id="info-loading" class="text-center py-2">
                                    <p class="text-gray-600 text-sm">Loading information...</p>
                                </div>
                                <div id="info-content" class="hidden">
                                    <div class="mb-2">
                                        <p class="text-sm font-medium text-gray-700">Description:</p>
                                        <p id="disease-description" class="text-gray-600"></p>
                                    </div>
                                    <div class="mb-2">
                                        <p class="text-sm font-medium text-gray-700">Symptoms:</p>
                                        <p id="disease-symptoms" class="text-gray-600"></p>
                                    </div>
                                    <div>
                                        <p class="text-sm font-medium text-gray-700">Treatment:</p>
                                        <p id="disease-treatment" class="text-gray-600"></p>
                                    </div>
                                </div>
                            </div>
                        </div>
                    </div>
                </div>
                
                <div class="mt-6">
                    <h3 class="text-lg font-semibold text-gray-800 mb-2">Other Possible Diseases</h3>
                    <div class="overflow-x-auto">
                        <table class="min-w-full divide-y divide-gray-200">
                            <thead class="bg-gray-50">
                                <tr>
                                    <th class="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Disease</th>
                                    <th class="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Confidence</th>
                                </tr>
                            </thead>
                            <tbody id="other-diseases" class="bg-white divide-y divide-gray-200">
                                <!-- Results will be inserted here -->
                            </tbody>
                        </table>
                    </div>
                </div>
            </div>
        </main>
    </div>

    <script>
        document.addEventListener('DOMContentLoaded', function() {
            const form = document.getElementById('upload-form');
            const imageUpload = document.getElementById('image-upload');
            const loading = document.getElementById('loading');
            const results = document.getElementById('results');
            const previewImage = document.getElementById('preview-image');
            const diseaseName = document.getElementById('disease-name');
            const confidence = document.getElementById('confidence');
            const otherDiseases = document.getElementById('other-diseases');
            const infoLoading = document.getElementById('info-loading');
            const infoContent = document.getElementById('info-content');
            const diseaseDescription = document.getElementById('disease-description');
            const diseaseSymptoms = document.getElementById('disease-symptoms');
            const diseaseTreatment = document.getElementById('disease-treatment');
            
            imageUpload.addEventListener('change', function(e) {
                const file = e.target.files[0];
                if (file) {
                    const reader = new FileReader();
                    reader.onload = function(e) {
                        previewImage.src = e.target.result;
                    };
                    reader.readAsDataURL(file);
                }
            });
            
            form.addEventListener('submit', function(e) {
                e.preventDefault();
                
                const formData = new FormData();
                const file = imageUpload.files[0];
                
                if (!file) {
                    alert('Please select an image file');
                    return;
                }
                
                formData.append('file', file);
                
                // Show loading spinner
                loading.style.display = 'block';
                results.style.display = 'none';
                
                // Send request to server
                fetch('/api/predict', {
                    method: 'POST',
                    body: formData
                })
                .then(response => response.json())
                .then(data => {
                    // Hide loading spinner
                    loading.style.display = 'none';
                    
                    if (data.error) {
                        alert('Error: ' + data.error);
                        return;
                    }
                    
                    // Update results
                    diseaseName.textContent = formatDiseaseName(data.class);
                    confidence.textContent = `Confidence: ${parseFloat(data.confidence).toFixed(2)}%`;
                    
                    // Clear previous results
                    otherDiseases.innerHTML = '';
                    
                    // Add other predictions
                    if (data.top_predictions && data.top_predictions.length > 1) {
                        data.top_predictions.slice(1).forEach(pred => {
                            const row = document.createElement('tr');
                            row.innerHTML = `
                                <td class="px-6 py-4 whitespace-nowrap">
                                    <div class="text-sm text-gray-900">${formatDiseaseName(pred.class)}</div>
                                </td>
                                <td class="px-6 py-4 whitespace-nowrap">
                                    <div class="text-sm text-gray-900">${parseFloat(pred.confidence).toFixed(2)}%</div>
                                </td>
                            `;
                            otherDiseases.appendChild(row);
                        });
                    }
                    
                    // Show results
                    results.style.display = 'block';
                    
                    // Fetch disease info
                    fetchDiseaseInfo(data.class);
                })
                .catch(error => {
                    loading.style.display = 'none';
                    alert('Error: ' + error.message);
                });
            });
            
            function fetchDiseaseInfo(diseaseName) {
                // Show loading for disease info
                infoLoading.style.display = 'block';
                infoContent.style.display = 'none';
                
                fetch('/api/disease-info', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json',
                    },
                    body: JSON.stringify({ disease: diseaseName })
                })
                .then(response => response.json())
                .then(data => {
                    // Hide loading
                    infoLoading.style.display = 'none';
                    
                    // Update info fields
                    diseaseDescription.textContent = data.description || 'Not available';
                    diseaseSymptoms.textContent = data.symptoms || 'Not available';
                    diseaseTreatment.textContent = data.treatment || 'Not available';
                    
                    // Show info content
                    infoContent.style.display = 'block';
                })
                .catch(error => {
                    infoLoading.style.display = 'none';
                    alert('Error fetching disease info: ' + error.message);
                });
            }
            
            function formatDiseaseName(name) {
                return name.replace(/_/g, ' ').replace(/__/g, ' - ');
            }
        });
    </script>
</body>
</html>"""
    return Response(html_content, mimetype='text/html')

@app.route('/api/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'})
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'})
    
    img_bytes = file.read()
    prediction = get_mock_prediction(img_bytes)
    return jsonify(prediction)

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

# This is needed for Vercel serverless functions
if __name__ == '__main__':
    app.run(debug=True)