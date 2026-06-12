import torch
import torch.nn.functional as F
from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles         
from fastapi.responses import FileResponse
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# 1. Initialize FastAPI
app = FastAPI(title="Hinglish Sentiment Analysis API", version="1.0")

# 2. Point to your local models folder relative to the src folder
MODEL_PATH = "../models/muril-hinglish"
id2label = {0: "Negative", 1: "Positive", 2: "Neutral"}

try:
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)
    model.eval()  # Set to evaluation mode (turns off dropout)
except Exception as e:
    raise RuntimeError(f"Failed to load model from {MODEL_PATH}. Error: {e}")

# 3. Request Schema
class CommentRequest(BaseModel):
    comment: str


# 4. Predict Endpoint
@app.post("/predict")
def predict_sentiment(payload: CommentRequest):
    if not payload.comment.strip():
        raise HTTPException(status_code=400, detail="Comment cannot be empty.")
    
    # Tokenize text
    inputs = tokenizer(
        payload.comment, 
        return_tensors="pt", 
        truncation=True, 
        padding="max_length", 
        max_length=128
    )
    
    # Run forward pass safely
    with torch.no_grad():
        outputs = model(**inputs)
        probabilities = F.softmax(outputs.logits, dim=-1).squeeze().tolist()
        prediction_id = torch.argmax(outputs.logits, dim=-1).item()
    
    # Return structured json
    return {
        "comment": payload.comment,
        "sentiment": id2label[prediction_id],
        "confidence_scores": {
            "Negative": round(probabilities[0], 4),
            "Positive": round(probabilities[1], 4),
            "Neutral": round(probabilities[2], 4)
        }
    }
# Mount the static folder directory to access asset pathways
app.mount("/static", StaticFiles(directory="static"), name="static")

# Catch-all index route to show the index.html page automatically
@app.get("/")
def serve_frontend():
    return FileResponse("static/index.html")
