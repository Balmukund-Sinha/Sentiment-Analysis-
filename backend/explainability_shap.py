
import torch
import shap
import numpy as np
from typing import List, Dict, Any
import logging
from functools import lru_cache

logger = logging.getLogger(__name__)

class ShapEngine:
    """
    Optimized SHAP Explainability Engine.
    Features: LRU Caching, Score Normalization, Top-K Filtering.
    """
    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer
        self.device = next(model.parameters()).device
        # Define prediction wrapper for SHAP
        def predictor(texts):
            # SHAP might pass strings or pre-tokenized items
            if not isinstance(texts, (list, np.ndarray)):
                texts = [texts]
            
            inputs = self.tokenizer(
                list(texts), 
                return_tensors="pt", 
                padding=True, 
                truncation=True, 
                max_length=512
            ).to(self.device)
            
            with torch.no_grad():
                outputs = self.model(**inputs)
                probs = torch.softmax(outputs.logits, dim=-1).cpu().numpy()
            return probs[:, 1] # Probability of Positive class

        # Use explicit Text masker
        masker = shap.maskers.Text(self.tokenizer)
        self.explainer = shap.Explainer(predictor, masker)

    @lru_cache(maxsize=100)
    def _get_shap_values(self, text: str):
        """Internal cached method for SHAP computation."""
        return self.explainer([text]) # Still use list for batch consistency in SHAP

    def explain(self, text: str, top_k: int = 0, normalize: bool = True) -> List[Dict[str, Any]]:
        """
        Explains the sentiment of a text using SHAP.
        Returns the FULL sequence of tokens with their importance scores in order.
        """
        try:
            shap_values = self._get_shap_values(text)
            
            # Extract tokens and values (in original sequence)
            tokens = shap_values.data[0]
            values = shap_values.values[0] # Expect (seq_len, 2) for binary classifier
            
            # SHAP returns (seq_len, num_classes). Select Positive class (index 1)
            if len(values.shape) == 2 and values.shape[1] > 1:
                values = values[:, 1]
            
            # Determine overall scale for normalization
            max_abs = max(np.abs(values)) if len(values) > 0 else 0
            
            explanation = []
            for token, val in zip(tokens, values):
                score = float(val)
                # Handle SentencePiece/DeBERTa space character (U+2581)
                clean_token = str(token).replace('\u2581', ' ')
                item = {
                    "token": clean_token,
                    "score": score
                }
                
                if normalize and max_abs > 0:
                    # Normalize to [-1, 1]
                    item["normalized_score"] = round(score / max_abs, 4)
                else:
                    item["normalized_score"] = score

                explanation.append(item)
            
            # Note: We NO LONGER SORT here to preserve sentence order for the UI
            return explanation

        except Exception as e:
            logger.error(f"SHAP explanation failed: {e}")
            return []
