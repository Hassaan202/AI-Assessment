from plant_disease_app import app

# This is the Vercel serverless function handler
def handler(request):
    return app(request)