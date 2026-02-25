
import os
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

# ==========================================
# Configuration
# ==========================================
MODEL_NAME = "roberta-base"  # SOTA upgrade from bert-base
MAX_LEN = 256
BATCH_SIZE = 16 # Effective batch size will be higher with gradient accumulation
GRADIENT_ACCUMULATION_STEPS = 2
EPOCHS = 4
LEARNING_RATE = 2e-5
OUTPUT_DIR = "./roberta_sentiment_model"

def compute_metrics(eval_pred):
    """
    Compute metrics for Trainer evaluation.
    """
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

def main():
    print(f"Initializing SOTA Training Pipeline with {MODEL_NAME}...")
    
    # Check device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    # 1. Load Dataset
    print("Loading IMDB dataset...")
    try:
        dataset = load_dataset("imdb", trust_remote_code=True)
    except Exception as e:
        print(f"Failed to load dataset: {e}")
        return

    # 2. Tokenizer
    print(f"Loading tokenizer: {MODEL_NAME}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    def tokenize_function(examples):
        return tokenizer(
            examples["text"], 
            padding="max_length", 
            truncation=True, 
            max_length=MAX_LEN
        )

    print("Tokenizing dataset (this might take a moment)...")
    tokenized_datasets = dataset.map(tokenize_function, batched=True)
    
    # Small subset for debugging if needed, but per requirements we use full
    # train_dataset = tokenized_datasets["train"].shuffle(seed=42).select(range(1000))
    # eval_dataset = tokenized_datasets["test"].shuffle(seed=42).select(range(500))
    
    train_dataset = tokenized_datasets["train"]
    eval_dataset = tokenized_datasets["test"]

    print(f"Train set size: {len(train_dataset)}")
    print(f"Eval set size: {len(eval_dataset)}")

    # 3. Model
    print("Initializing Model...")
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME, 
        num_labels=2
    )

    # 4. Training Arguments
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        learning_rate=LEARNING_RATE,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE * 2,
        num_train_epochs=EPOCHS,
        weight_decay=0.01,
        gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
        warmup_ratio=0.1,
        # fp16=True, # Enable Mixed Precision if GPU supports it
        fp16=torch.cuda.is_available(), 
        logging_dir='./logs',
        logging_steps=100,
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
        save_total_limit=2,
        report_to="none" # Disable wandb/mlflow for this standalone script
    )

    # 5. Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=2)]
    )

    # 6. Train
    print("Starting Training...")
    trainer.train()

    # 7. Final Evaluation
    print("Evaluating...")
    eval_results = trainer.evaluate()
    print(f"Evaluation Results: {eval_results}")

    # 8. Save
    print(f"Saving model to {OUTPUT_DIR}...")
    trainer.save_model(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    print("Done!")

if __name__ == "__main__":
    main()
