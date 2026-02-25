
import os
import time
import logging
import sys
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any

# Ensure local imports work
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import modular components
from engine import InferenceEngine

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Production Sentiment API",
    description="High-performance, explainable sentiment analysis using DeBERTa-v3.",
    version="4.0.0"
)

# CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize Engine with absolute path detection
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATHS = [
    os.path.join(ROOT_DIR, "deberta_sentiment_model"),
    os.path.join(ROOT_DIR, "roberta_sentiment_model"),
    os.path.join(ROOT_DIR, "saved_model"),
    "./deberta_sentiment_model",
    "./roberta_sentiment_model",
    "./saved_model"
]
engine = InferenceEngine(MODEL_PATHS)

# Explainability Fallback Logic
try:
    from explainability_shap import ShapEngine
    explainer = ShapEngine(engine.model, engine.tokenizer)
    logger.info("SHAP Explainability Engine loaded.")
except Exception as e:
    logger.warning(f"SHAP failed to load, falling back to Attention basics: {e}")
    try:
        from explainability import ExplainabilityEngine
        explainer = ExplainabilityEngine(engine.model, engine.tokenizer, engine.device)
        logger.info("Attention-based Explainability Engine loaded.")
    except Exception as e2:
        logger.error(f"All explainability engines failed: {e2}")
        explainer = None

# --- Schemas ---

class PredictRequest(BaseModel):
    text: str = Field(..., min_length=1, example="I absolutely loved this movie!")
    explain: bool = False
    confidence_threshold: float = 0.7

class BatchPredictRequest(BaseModel):
    texts: List[str] = Field(..., min_items=1, example=["Great movie!", "Waste of time."])
    explain: bool = False
    confidence_threshold: float = 0.7

class PredictResponse(BaseModel):
    sentiment: str
    confidence: float
    status: str
    probabilities: Dict[str, float]
    model_version: str
    explanation: Optional[List[Dict[str, Any]]] = None
    latency_ms: float

# --- Endpoints ---

@app.post("/predict", response_model=PredictResponse)
async def predict(request: PredictRequest):
    start_time = time.time()
    
    try:
        # 1. Inference
        results = engine.predict(request.text, request.confidence_threshold)
        res = results[0]
        
        # 2. Explain (Optional)
        explanation = None
        if request.explain and explainer:
            explanation = explainer.explain(request.text)
            
        latency = (time.time() - start_time) * 1000
        
        return {
            **res,
            "explanation": explanation,
            "latency_ms": round(latency, 2)
        }
        
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/predict_batch", response_model=List[PredictResponse])
async def predict_batch(request: BatchPredictRequest):
    start_time = time.time()
    
    try:
        # 1. True Batch Inference (Optimized via Engine)
        results = engine.predict(request.texts, request.confidence_threshold)
        
        # 2. Sequential Explain (Explainability is typically slow)
        final_results = []
        for i, res in enumerate(results):
            explanation = None
            if request.explain and explainer:
                explanation = explainer.explain(request.texts[i])
            
            final_results.append({
                **res,
                "explanation": explanation,
                "latency_ms": round((time.time() - start_time) * 1000, 2)
            })
            
        return final_results
        
    except Exception as e:
        logger.error(f"Batch prediction error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health():
    return {
        "status": "healthy",
        "model": engine.model_version,
        "device": str(engine.device),
        "bf16_enabled": engine.use_bf16,
        "torch_compiled": hasattr(engine.model, "_orig_mod") or hasattr(engine.model, "forward") # Simplified check
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
