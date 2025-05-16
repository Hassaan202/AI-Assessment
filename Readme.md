# Plant Disease Classifier

A web-based application that uses AI to identify plant diseases from leaf images. The application combines PyTorch for image classification and Google's Gemini AI for providing detailed disease information.

> **Note**: This application was developed using AI tools, including Cursor and Claude AI, to demonstrate the capabilities of AI-assisted software development.

## Dataset and Model

The classification model was trained on the PlantVillage dataset, which is a large-scale dataset of plant leaf images for disease detection. The dataset was originally published in:

```
@article{Mohanty_Hughes_Salathé_2016,
title={Using deep learning for image-based plant disease detection},
volume={7},
DOI={10.3389/fpls.2016.01419},
journal={Frontiers in Plant Science},
author={Mohanty, Sharada P. and Hughes, David P. and Salathé, Marcel},
year={2016},
month={Sep}
}
```

The dataset contains over 54,000 images of plant leaves, covering 38 different classes of plant diseases across 14 crop species. This extensive dataset enables the model to accurately identify various plant diseases with high confidence.

## Features

- **Image Classification**: Upload images of plant leaves to identify diseases
- **AI-Powered Analysis**: Uses PyTorch model for accurate disease detection
- **Detailed Information**: Integrates with Gemini AI to provide comprehensive disease information
- **User-Friendly Interface**: Modern, responsive design with intuitive controls
- **Real-time Feedback**: Immediate analysis results with confidence scores
- **Top 5 Predictions**: Shows multiple possible matches with confidence levels
- **Interactive UI**: Drag-and-drop support and image preview
- **Detailed Instructions**: Collapsible panel with usage guidelines and best practices

## Supported Plants

Currently, the application can identify diseases in:
- Tomato plants
- Potato plants
- Pepper plants

## Prerequisites

- Python 3.8 or higher
- pip (Python package installer)
- Google Cloud account with Gemini API access

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd plant-disease-classifier
```

2. Create a virtual environment (recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install required packages:
```bash
pip install -r requirements.txt
```

4. Set up environment variables:
   - Create a `.env` file in the project root
   - Add your Gemini API key:
   ```
   GEMINI_API_KEY=your_api_key_here
   ```

## Project Structure

```
plant-disease-classifier/
├── app.py              # Main Flask application
├── requirements.txt    # Python dependencies
├── .env               # Environment variables (create this)
├── templates/
│   └── index.html     # Frontend template
└── static/
    └── model/         # PyTorch model files
```

## Usage

1. Start the Flask server:
```bash
python app.py
```

2. Open your web browser and navigate to:
```
http://localhost:5000
```

3. Upload an image:
   - Click "Browse Files" or drag and drop an image
   - Supported formats: JPG, PNG, JPEG
   - For best results, use clear, well-lit images of single leaves

4. View results:
   - The application will display the detected disease
   - Confidence score and top 5 predictions
   - Detailed information about the disease
   - Treatment recommendations

## Deployment to Vercel

1. **Prepare Your Project**:
   - Make sure all your files are committed to a GitHub repository
   - Ensure your `requirements.txt` is up to date
   - Create a `vercel.json` file in your project root with the following content:
   ```json
   {
     "version": 2,
     "builds": [
       {
         "src": "app.py",
         "use": "@vercel/python"
       }
     ],
     "routes": [
       {
         "src": "/(.*)",
         "dest": "app.py"
       }
     ]
   }
   ```

2. **Set Up Vercel**:
   - Create an account on [Vercel](https://vercel.com) if you haven't already
   - Install Vercel CLI: `npm i -g vercel`
   - Login to Vercel: `vercel login`

3. **Deploy**:
   - Navigate to your project directory
   - Run `vercel` to deploy
   - Follow the prompts to:
     - Link to your Vercel account
     - Select your project
     - Configure project settings

4. **Environment Variables**:
   - In the Vercel dashboard, go to your project settings
   - Navigate to the "Environment Variables" section
   - Add your `GEMINI_API_KEY` and other sensitive variables

5. **Automatic Deployments**:
   - Vercel will automatically deploy when you push to your GitHub repository
   - Each push to the main branch creates a new deployment
   - You can view deployment status and logs in the Vercel dashboard

6. **Custom Domain** (Optional):
   - In the Vercel dashboard, go to your project settings
   - Navigate to the "Domains" section
   - Add and configure your custom domain

> **Note**: Make sure your model files are properly included in your repository and the paths in your code match the deployment structure.

## Best Practices for Image Upload

For optimal results, follow these guidelines when taking photos:
- Use clear, well-lit photos
- Focus on a single leaf (preferred) or small group of affected leaves
- Use higher resolution images
- Ensure a neutral or contrasting background
- Take photos from directly above the leaf

## Dependencies

- Flask: Web framework
- PyTorch: Deep learning framework
- google-generativeai: Gemini AI integration
- python-dotenv: Environment variable management
- Pillow: Image processing
- Other dependencies listed in requirements.txt

## Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## AI-Assisted Development

This project was developed using AI tools to demonstrate modern software development practices:

- **Code Generation**: Initial code structure and implementation was assisted by AI
- **UI Design**: Frontend design and styling was created with AI assistance
- **Documentation**: This README and inline documentation were generated with AI
- **Debugging**: AI tools helped identify and fix issues during development
- **Best Practices**: AI assisted in implementing coding standards and patterns

The use of AI in development helped accelerate the creation process while maintaining code quality and following best practices.

## Acknowledgments

- PyTorch team for the deep learning framework
- Google for the Gemini AI API
- Flask team for the web framework
- PlantVillage dataset creators for providing the training data
- AI tools that assisted in the development process