import google.generativeai as genai
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()

# Configure the API
genai.configure(api_key=os.getenv('GOOGLE_API_KEY'))

# List all available models
for m in genai.list_models():
    if 'generateContent' in m.supported_generation_methods:
        print(f"Model: {m.name}")
        print(f"Display name: {m.display_name}")
        print(f"Description: {m.description}")
        print(f"Generation methods: {m.supported_generation_methods}")
        print("-" * 50) 