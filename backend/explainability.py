
import torch
import numpy as np

class ExplainabilityEngine:
    """
    Provides explanation for model predictions.
    Currently implements Attention-based importance.
    """
    def __init__(self, model, tokenizer, device):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        
    def explain(self, text: str, max_len: int = 512) -> List[Dict[str, Any]]:
        """
        Standardized explain method for attention-based importance.
        """
        try:
            self.model.eval()
            
            # Tokenize
            inputs = self.tokenizer(
                text, 
                return_tensors="pt", 
                truncation=True, 
                max_length=max_len
            ).to(self.device)
            
            input_ids = inputs['input_ids']
            
            with torch.no_grad():
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=inputs['attention_mask'],
                    output_attentions=True
                )
                
            # Average attention across heads of the last layer
            last_layer_attn = outputs.attentions[-1] 
            avg_attn = last_layer_attn.mean(dim=1) 
            cls_attn = avg_attn[0, 0, :].cpu().numpy() 
            
            # Normalize to [0, 1] for attention (attention is always positive)
            max_val = cls_attn.max()
            min_val = cls_attn.min()
            norm_attn = (cls_attn - min_val) / (max_val - min_val + 1e-9)
            
            # Decode tokens
            tokens = self.tokenizer.convert_ids_to_tokens(input_ids[0])
            
            explanation = []
            for token, score, norm_score in zip(tokens, cls_attn, norm_attn):
                 # Filter specials
                 if token in [self.tokenizer.cls_token, self.tokenizer.sep_token, self.tokenizer.pad_token]:
                     continue
                     
                 # Handle SentencePiece/DeBERTa spaces (U+2581)
                 clean_token = token.replace('\u2581', ' ')
                 
                 explanation.append({
                     "token": clean_token, 
                     "score": float(score),
                     "normalized_score": float(norm_score)
                 })
                 
            return explanation
        except Exception as e:
            print(f"Attention explain failed: {e}")
            return []
