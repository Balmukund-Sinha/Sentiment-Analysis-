
import os
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from backend.explainability_shap import ShapEngine

# Setup paths
MODEL_PATH = "deberta_sentiment_model"
if not os.path.exists(MODEL_PATH):
    MODEL_PATH = "../deberta_sentiment_model"

print(f"Loading model from {MODEL_PATH}")
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)

engine = ShapEngine(model, tokenizer)
text = "This movie was absolutely amazing!"
print(f"Explaining: '{text}'")
try:
    explanation = engine.explain(text)
    print(f"Explanation length: {len(explanation)}")
    if len(explanation) > 0:
        print(f"First 3 items: {explanation[:3]}")
    else:
        print("WARNING: Explanation is EMPTY!")
except Exception as e:
    print(f"SHAP Error in test: {e}")
    import traceback
    traceback.print_exc()
