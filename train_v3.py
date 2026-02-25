
import os
import sys
import torch
import numpy as np
from datasets import load_dataset
from transformers import (
    AutoTokenizer, 
    AutoModelForSequenceClassification, 
    TrainingArguments, 
    Trainer,
    EarlyStoppingCallback
)
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

def check_environment():
    """
    Diagnose environment for the user.
    """
    print("="*40)
    print("      ENVIRONMENT DIAGNOSTICS      ")
    print("="*40)
    print(f"Python: {sys.version.split()[0]}")
    print(f"PyTorch Version: {torch.__version__}")
    
    cuda_available = torch.cuda.is_available()
    print(f"CUDA Available: {cuda_available}")
    
    if cuda_available:
        print(f"Device Name: {torch.cuda.get_device_name(0)}")
        print(f"CUDA Version: {torch.version.cuda}")
        print(f"CUDNN Version: {torch.backends.cudnn.version()}")
    else:
        print("WARNING: CUDA IS NOT AVAILABLE. TRAINING WILL BE SLOW (CPU).")
        print("Please run: pip install --pre torch --index-url https://download.pytorch.org/whl/nightly/cu124")
        
    print("="*40)
    return cuda_available

# ==========================================
# Configuration (RTX 4060 Optimized)
# ==========================================
MODEL_NAME = "microsoft/deberta-v3-base" 
MAX_LEN = 512 
BATCH_SIZE = 16 # Increased from 8 (Safe with Gradient Checkpointing on 8GB VRAM)
GRADIENT_ACCUMULATION_STEPS = 2 # Adjusted for larger batch size (Total effective = 32)
EPOCHS = 5
LEARNING_RATE = 2e-5
OUTPUT_DIR = "./deberta_sentiment_model"
WEIGHT_DECAY = 0.01
LR_DECAY = 0.9

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, predictions, average='binary')
    acc = accuracy_score(labels, predictions)
    return {
        'accuracy': acc,
        'f1': f1,
        'precision': precision,
        'recall': recall
    }

class CustomTrainer(Trainer):
    def create_optimizer(self):
        """
        Layer-wise Learning Rate Decay (LLRD) optimizer.
        """
        opt_model = self.model
        
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in opt_model.named_parameters() if "classifier" in n or "pooler" in n],
                "weight_decay": 0.0,
                "lr": self.args.learning_rate,
            },
        ]

        if hasattr(opt_model, "deberta"):
            layers = [opt_model.deberta.embeddings] + list(opt_model.deberta.encoder.layer)
        else:
            # Fallback for RoBERTa/BERT if changed
            layers = [opt_model.base_model.embeddings] + list(opt_model.base_model.encoder.layer)
            
        layers.reverse()
        
        lr = self.args.learning_rate
        for layer in layers:
            lr *= LR_DECAY
            optimizer_grouped_parameters += [
                {
                    "params": [p for n, p in layer.named_parameters() if not any(nd in n for nd in no_decay)],
                    "weight_decay": self.args.weight_decay,
                    "lr": lr,
                },
                {
                    "params": [p for n, p in layer.named_parameters() if any(nd in n for nd in no_decay)],
                    "weight_decay": 0.0,
                    "lr": lr,
                },
            ]
            
        self.optimizer = torch.optim.AdamW(optimizer_grouped_parameters)
        return self.optimizer

def main():
    check_environment()
    # Resume strictly if check passes
    print(f"Initializing Phase 3 Pipeline: {MODEL_NAME}...")
    
    # Load Dataset
    try:
        dataset = load_dataset("imdb")
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return

    # Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=False)

    def tokenize_function(examples):
        return tokenizer(
            examples["text"], 
            padding="max_length", 
            truncation=True, 
            max_length=MAX_LEN
        )

    print("Tokenizing...")
    tokenized_datasets = dataset.map(tokenize_function, batched=True)
    train_dataset = tokenized_datasets["train"]
    eval_dataset = tokenized_datasets["test"]

    # Model
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=2)

    # Arguments
    # Optimizations for RTX 4060 (Ada Lovelace Architecture):
    # 1. bf16=True: Brain Float 16 is supported on RTX 30/40 series, better stability than fp16
    # 2. tf32=True: TensorFloat-32 for faster FP32 math on Ampere/Ada
    # 3. gradient_checkpointing=True: Trades compute for VRAM, allowing larger batch size (16 vs 8)
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        eval_strategy="epoch",
        save_strategy="epoch",
        learning_rate=LEARNING_RATE,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE * 2,
        num_train_epochs=EPOCHS,
        weight_decay=WEIGHT_DECAY,
        gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
        warmup_ratio=0.1,
        
        # GPU Optimizations
        fp16=False,    # Disable FP16
        bf16=True,     # Enable BF16 (Better for DeBERTa on RTX 4060)
        tf32=True,     # Enable TF32 (Ampere/Ada feature)
        gradient_checkpointing=True, # Saves VRAM -> Larger Batch Size
        gradient_checkpointing_kwargs={"use_reentrant": False}, # Fixes "backward through graph a second time" error
        
        dataloader_num_workers=0, # Keep 0 for Windows safety
        dataloader_pin_memory=True, # Optimized data transfer
        logging_steps=50,
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
        save_total_limit=1,
        report_to="none"
    )

    # Trainer
    trainer = CustomTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=2)]
    )

    print("Starting Training (DeBERTa-v3)...")
    trainer.train()

    print("Evaluating...")
    eval_results = trainer.evaluate()
    print(f"Results: {eval_results}")

    print(f"Saving to {OUTPUT_DIR}...")
    trainer.save_model(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)

if __name__ == "__main__":
    main()
