
import os
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import logging
from typing import List, Dict, Any, Union

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class InferenceEngine:
    """
    Production-grade Inference Engine for Transformers.
    Features: torch.compile, Mixed Precision (BF16), Dynamic Padding, Batching.
    """
    def __init__(self, model_paths: List[str]):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None
        self.tokenizer = None
        self.model_version = "Unknown"
        self._load_model(model_paths)
        
        # Optimization: Mixed Precision
        # RTX 4060 supports bfloat16 (better stability than float16)
        self.use_bf16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
        self.dtype = torch.bfloat16 if self.use_bf16 else torch.float32
        
        if self.model:
            # Move to device and set to eval mode
            self.model.to(self.device).eval()
            
            # Optimization: torch.compile (PyTorch 2.0+)
            # Note: Triton (required for compile) is often unavailable on Windows.
            if os.name != "nt": 
                try:
                    logger.info("Applying torch.compile() for inference optimization...")
                    self.model = torch.compile(self.model)
                    logger.info("torch.compile() successful.")
                except Exception as e:
                    logger.warning(f"torch.compile() failed: {e}")
            else:
                logger.info("Skipping torch.compile() on Windows (Triton dependency). Using standard eager mode.")

    def _load_model(self, paths: List[str]):
        """Iteratively try to load the best available model."""
        for path in paths:
            abs_path = os.path.abspath(path)
            if os.path.exists(path):
                try:
                    logger.info(f"Attempting to load model from {path}...")
                    self.tokenizer = AutoTokenizer.from_pretrained(path)
                    self.model = AutoModelForSequenceClassification.from_pretrained(path)
                    
                    if "deberta" in path.lower():
                        self.model_version = "DeBERTa-v3"
                    elif "roberta" in path.lower():
                        self.model_version = "RoBERTa"
                    else:
                        self.model_version = "BERT-Base"
                        
                    logger.info(f"Successfully loaded {self.model_version} from {path}")
                    return
                except Exception as e:
                    logger.error(f"Failed to load from {path}: {e}")
        
        logger.critical("No valid models found in any provided paths!")

    @torch.inference_mode()
    def predict(self, texts: Union[str, List[str]], confidence_threshold: float = 0.7, batch_size: int = 32) -> List[Dict[str, Any]]:
        """
        Unified prediction method for single or batch inputs.
        Supports: Chunked processing (to prevent OOM), Dynamic padding, Mixed Precision.
        """
        if isinstance(texts, str):
            texts = [texts]

        if not self.model or not self.tokenizer:
            return [{"error": "Model not loaded"}] * len(texts)

        all_results = []
        
        # Process in chunks to prevent GPU OOM for very large batch requests
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]
            
            # Dynamic Padding: Tokens are padded to the longest in THIS chunk
            inputs = self.tokenizer(
                batch_texts, 
                padding=True, 
                truncation=True, 
                max_length=512, 
                return_tensors="pt"
            ).to(self.device)

            # Mixed Precision Inference
            with torch.autocast(device_type=self.device.type, dtype=self.dtype):
                outputs = self.model(**inputs)
                logits = outputs.logits
                
            # Calibration & Softmax
            probs = F.softmax(logits, dim=-1)
            
            for j in range(len(batch_texts)):
                conf, label_idx = torch.max(probs[j], dim=-1)
                conf_val = conf.item()
                label = "Positive" if label_idx.item() == 1 else "Negative"
                
                # Confidence Thresholding (Uncertainty Detection)
                status = "Reliable" if conf_val >= confidence_threshold else "Uncertain"
                
                all_results.append({
                    "sentiment": label,
                    "confidence": round(conf_val, 4),
                    "status": status,
                    "probabilities": {
                        "negative": round(probs[j][0].item(), 4),
                        "positive": round(probs[j][1].item(), 4)
                    },
                    "model_version": self.model_version
                })
            
        return all_results
