import streamlit as st
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import io
import os

# Page configuration
st.set_page_config(
    page_title="Plant Disease Detection",
    page_icon="🌿",
    layout="wide"
)

# Streamlined UI with minimal styling
st.markdown("""
    <style>
    /* Base styling */
    body {
        font-family: 'Roboto', sans-serif;
    }
    
    /* Background */
    .main, .stApp {
        background-color: #e8f5e9;
    }
    
    /* Header styling */
    h1 {
        color: #ffffff;
        font-weight: 700;
        background-color: #2e7d32;
        padding: 1rem;
        border-radius: 8px;
        text-align: center;
        margin-bottom: 1.5rem;
    }
    
    h2, h3 {
        color: white;
        background-color: #2e7d32;
        padding: 0.5rem 1rem;
        border-radius: 4px;
        font-weight: 600;
    }
    
    /* Button styling */
    .stButton > button {
        background-color: #2e7d32;
        color: white;
        font-weight: 600;
        border-radius: 4px;
        border: none;
        width: 100%;
        transition: all 0.2s ease;
    }
    .stButton > button:hover {
        background-color: #1b5e20;
        box-shadow: 0 2px 5px rgba(0,0,0,0.2);
    }
    
    /* Collapsible sections */
    .streamlit-expanderHeader {
        background-color: #c8e6c9;
        border-radius: 4px;
        padding: 0.5rem;
        font-weight: 600;
    }
    
    /* Results styling */
    .result-section {
        background-color: white;
        padding: 1.5rem;
        border-radius: 8px;
        margin-top: 1rem;
        box-shadow: 0 2px 5px rgba(0,0,0,0.1);
    }
    
    /* Prediction display */
    .prediction {
        background-color: #f1f8e9;
        padding: 1rem;
        border-radius: 4px;
        border-left: 4px solid #2e7d32;
    }
    
    /* Confidence meter */
    .confidence-meter {
        height: 8px;
        background-color: #e0e0e0;
        border-radius: 4px;
        margin: 8px 0;
    }
    
    .confidence-value {
        height: 100%;
        background-color: #2e7d32;
        border-radius: 4px;
    }
    
    /* Hide empty whitespace */
    .element-container:empty {
        display: none;
    }
    
    /* Apply custom styling to file uploader */
    .stFileUploader {
        padding: 0.5rem;
    }
    
    /* Footer */
    .footer {
        text-align: center;
        margin-top: 1rem;
        font-size: 0.8rem;
        color: #666;
    }
    </style>
""", unsafe_allow_html=True)

# Define the CNN model architecture
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

# Class names for prediction
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

# Treatment recommendations for each class
disease_info = {
    'Pepper__bell___Bacterial_spot': {
        'description': 'Bacterial spot causes small, dark, raised spots on pepper leaves and fruits.',
        'symptoms': 'Brown to black circular spots on leaves, stems, and fruits. Spots may merge as infection progresses.',
        'treatment': 'Apply copper-based bactericides, ensure proper plant spacing for airflow, and practice crop rotation.'
    },
    'Potato___healthy': {
        'description': 'This plant shows no signs of disease and appears healthy.',
        'symptoms': 'No visible symptoms of disease. Leaves show normal coloration and structure.',
        'treatment': 'Continue regular maintenance and preventive measures.'
    },
    # Add more disease information for other classes as needed
}

# Fill in missing disease information with generic content
for class_name in class_names:
    if class_name not in disease_info:
        disease_info[class_name] = {
            'description': f'A disease affecting {class_name.split("_")[0]} plants.',
            'symptoms': 'Symptoms may include spots on leaves, wilting, discoloration, or unusual growths.',
            'treatment': 'Consult with an agricultural expert for proper diagnosis and treatment recommendations.'
        }

@st.cache_resource
def load_model():
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Initialize model
    model = PlantDiseaseCNN(len(class_names))
    
    # Load trained weights
    try:
        model_path = 'plant_model_improved.pth'
        if os.path.exists(model_path):
            model.load_state_dict(torch.load(model_path, map_location=device))
            model.eval()
            return model, device
        else:
            st.error(f"Model file not found at {model_path}")
            return None, device
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None, device

def preprocess_image(image):
    transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                           std=[0.229, 0.224, 0.225])
    ])
    return transform(image).unsqueeze(0)

def predict_disease(model, image, device):
    # Preprocess image
    input_tensor = preprocess_image(image).to(device)
    
    # Get prediction
    with torch.no_grad():
        outputs = model(input_tensor)
        probabilities = torch.nn.functional.softmax(outputs, dim=1)
        
        # Get top 3 predictions
        probs, classes = torch.topk(probabilities, 3)
        
        results = []
        for i in range(3):
            results.append({
                'class': class_names[classes[0][i]],
                'probability': probs[0][i].item()
            })
        
    return results

def main():
    # Header
    st.markdown("<h1>🌿 Plant Disease Detection</h1>", unsafe_allow_html=True)
    
    # Load model
    model, device = load_model()
    if model is None:
        st.error("Failed to load model. Please ensure the model file exists.")
        return
    
    # Togglable instructions section
    with st.expander("📋 How to use this app"):
        st.markdown("""
            ### Instructions
            1. Upload a clear, well-lit photo of a **single plant leaf**
            2. Make sure the leaf fills most of the frame
            3. Avoid blurry images or images with multiple leaves
            4. Click 'Analyze Image' to get results
            
            For best results, ensure the image shows any symptoms clearly.
        """)
    
    # Image upload - using native streamlit uploader without custom container
    st.subheader("Upload Plant Leaf Image")
    uploaded_file = st.file_uploader("", type=["jpg", "jpeg", "png"])
    
    if uploaded_file is not None:
        # Display the uploaded image
        image = Image.open(uploaded_file).convert('RGB')
        st.image(image, use_container_width=True)
        
        # Analyze button
        analyze_button = st.button("🔍 Analyze Image", use_container_width=True)
        
        # Results section displayed directly below the image and button
        if analyze_button:
            with st.spinner("Analyzing image..."):
                results = predict_disease(model, image, device)
                
                st.markdown("<div class='result-section'>", unsafe_allow_html=True)
                
                # Display top prediction
                top_result = results[0]
                disease_name = top_result['class'].replace('_', ' ')
                confidence = top_result['probability'] * 100
                
                st.markdown(f"<h2>Prediction Results</h2>", unsafe_allow_html=True)
                st.markdown(f"""
                    <div class='prediction'>
                        <p style='font-size: 1.2rem; font-weight: 600; color: #2e7d32;'>{disease_name}</p>
                        <p>Confidence: {confidence:.1f}%</p>
                        <div class='confidence-meter'>
                            <div class='confidence-value' style='width: {confidence}%;'></div>
                        </div>
                    </div>
                """, unsafe_allow_html=True)
                
                # Disease information for top result
                disease_data = disease_info.get(top_result['class'], {
                    'description': 'Information not available',
                    'symptoms': 'Information not available',
                    'treatment': 'Consult with an agricultural expert for diagnosis and treatment'
                })
                
                # Create tabs for different sections of information
                tab1, tab2, tab3 = st.tabs(["Description", "Symptoms", "Treatment"])
                
                with tab1:
                    st.write(disease_data['description'])
                
                with tab2:
                    st.write(disease_data['symptoms'])
                
                with tab3:
                    st.write(disease_data['treatment'])
                
                # Alternative possibilities in expandable section
                with st.expander("Alternative Possibilities"):
                    for i in range(1, len(results)):
                        alt = results[i]
                        alt_name = alt['class'].replace('_', ' ')
                        alt_conf = alt['probability'] * 100
                        
                        st.markdown(f"""
                            <div style='padding: 0.5rem; border-bottom: 1px solid #e0e0e0;'>
                                <p><strong>{alt_name}</strong> ({alt_conf:.1f}%)</p>
                            </div>
                        """, unsafe_allow_html=True)
                
                st.markdown("</div>", unsafe_allow_html=True)
    
    else:
        # Simple message when no image is uploaded
        st.info("Please upload an image of a plant leaf to detect diseases")
    
    # Footer
    st.markdown("""
        <div class='footer'>
            Powered by AI and Computer Vision | Plant Disease Detection System
        </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()