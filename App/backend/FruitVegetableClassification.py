from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
import tensorflow as tf
import numpy as np
from PIL import Image 
from io import BytesIO
import uvicorn
from fastapi.middleware.cors import CORSMiddleware
from ultralytics import YOLO
import os

app = FastAPI()

#app.mount("/static", StaticFiles(directory="static"), name="static")
base_dir = os.path.dirname(os.path.abspath(__file__))  # Directory of the current script
frontend_dir = os.path.join(base_dir, "../frontend")  # Adjust path as needed

# Sets the templates directory to the `build` folder from `npm run build`
# this is where you'll find the index.html file.
templates = Jinja2Templates(directory=frontend_dir)

# Mounts the `static` folder within the `build` folder to the `/static` route.
app.mount('/static', StaticFiles(directory=os.path.join(frontend_dir, "static")), 'static')

@app.get("/")
def serve_react():
    return FileResponse(os.path.join(frontend_dir, "index.html"))

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],   # Allows all methods (GET, POST, etc.)
    allow_headers=["*"],   # Allows all headers
)

class_names = ['Apple__Healthy', 'Apple__Rotten', 'Banana__Healthy', 'Banana__Rotten', 'Bellpepper__Healthy', 'Bellpepper__Rotten', 'Carrot__Healthy', 'Carrot__Rotten', 'Cucumber__Healthy', 'Cucumber__Rotten', 'Grape__Healthy', 'Grape__Rotten', 'Guava__Healthy', 'Guava__Rotten', 'Jujube__Healthy', 'Jujube__Rotten', 'Mango__Healthy', 'Mango__Rotten', 'Orange__Healthy', 'Orange__Rotten', 'Pomegranate__Healthy', 'Pomegranate__Rotten', 'Potato__Healthy', 'Potato__Rotten', 'Strawberry__Healthy', 'Strawberry__Rotten', 'Tomato__Healthy', 'Tomato__Rotten'] 

yolo_class_names = ['Healthy', 'Rotten']

fruit_names = ['Apple', 'Banana', 'Cucumber', 'Grape', 'Guava', 'Jujube', 'Mango', 'Orange', 'Pomegranate', 'Strawberry', 'Tomato']
vegetable_names = ['Bellpepper', 'Carrot', 'Potato']

class PredictionResponse(BaseModel):    
    predicted_type: str
    predicted_name: str
    confidence: float
    predicted_class: str    
 
# Load models once at startup
model_paths = {
    "EfficientNet": os.path.join(base_dir, "../models/efficientnet.keras"),
    "MobileNet": os.path.join(base_dir, "../models/mobilenet.keras"),
    "ResNet": os.path.join(base_dir, "../models/resnet.keras"),
    "YOLO": os.path.join(base_dir, "../models/yolo_fruit_veg_optimized.pt")
}

models = {}

def load_model(model_name: str):
    if model_name not in model_paths.keys():
        print(f"Model {model_name} not found")
        raise HTTPException(status_code=400, detail=f"Model '{model_name}' not found. Please provide a valid model name.")
    if model_name not in models:
        models[model_name] = tf.keras.models.load_model(model_paths[model_name]) if model_name != "YOLO" else YOLO(model_paths[model_name])
    return models[model_name]
    
def preprocess_image(image_bytes: bytes, model_name: str) -> np.ndarray:
    # Load the image from bytes
    image = Image.open(BytesIO(image_bytes))
    
    if model_name == "YOLO":
        return image
    
    # Save the image as JPEG to a buffer
    buffer = BytesIO()
    image.save(buffer, format="JPEG")
    buffer.seek(0)
    
    # Re-load the JPEG image from the buffer and convert it to RGB
    jpeg_image = Image.open(buffer).convert("RGB")
        
    # Resize the image to 224x224
    jpeg_image = jpeg_image.resize((224, 224))
        
    # Convert to NumPy array
    image_array = np.array(jpeg_image)
    
    # Add batch dimension (for model input)
    image_array = np.expand_dims(image_array, axis=0)
    
    return image_array

@app.post("/predict", response_model=PredictionResponse)
async def predict(model_name: str, file: UploadFile = File(...)):    
    # Load the correct model based on the model_name in the request
    model = load_model(model_name)
    
    # Read image file
    image_bytes = await file.read()
    
    # Preprocess image
    preprocessed_image = preprocess_image(image_bytes, model_name)
    
    if model_name == "YOLO":
        result = model(preprocessed_image)  # Perform inference on the image
        predictions = result[0].boxes
        
        if len(predictions) == 0:
            return PredictionResponse(predicted_type="NaN", predicted_name="NaN", confidence=0.0, predicted_class="Unknown")
        
        confidence = predictions[0].conf[0]  # Confidence score
        class_id = int(predictions[0].cls[0])  # Class ID
        
        if(confidence < 0.5):
            return PredictionResponse(predicted_type="NaN", predicted_name="NaN", confidence=confidence, predicted_class="Unknown")
    
        return PredictionResponse(predicted_type="NaN", predicted_name="NaN", confidence=confidence, predicted_class=yolo_class_names[class_id])
    
    # Make predictions
    predictions = model.predict(preprocessed_image)
    
    # Get the highest confidence score and corresponding class
    predicted_idx = np.argmax(predictions[0])
    confidence = predictions[0][predicted_idx]
    predicted_class = class_names[predicted_idx]
    
    predicted_name_class = predicted_class.split("__")
    
    if(confidence < 0.5):
        return PredictionResponse(predicted_type="Unknown", predicted_name="Unknown", confidence=float(confidence), predicted_class="Unknown")
    
    return PredictionResponse(predicted_type="Vegetable" if predicted_name_class[0] in vegetable_names else "Fruit", predicted_name=predicted_name_class[0], confidence=float(confidence), predicted_class=predicted_name_class[1])

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=9000)