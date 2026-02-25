
import os
import re
import random
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizerFast, BertForSequenceClassification, AdamW, get_linear_schedule_with_warmup
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from datasets import load_dataset
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")

# Set random seeds for reproducibility
def set_seed(seed_val=42):
    random.seed(seed_val)
    np.random.seed(seed_val)
    torch.manual_seed(seed_val)
    torch.cuda.manual_seed_all(seed_val)

set_seed(42)

# ==========================================
# Configuration
# ==========================================
class Config:
    MAX_LEN = 256
    BATCH_SIZE = 16  # Reduced for standard GPU memory safety, increase if memory allows
    EPOCHS = 3
    LEARNING_RATE = 2e-5
    MODEL_NAME = 'bert-base-uncased'
    DATA_SAMPLE_SIZE = 10000 # Use 10000 samples as requested (or set to None for full dataset)
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

print(f"Using device: {Config.DEVICE}")

# ==========================================
# 1. Data Preprocessing & Loading
# ==========================================
def clean_text(text):
    """
    Simple preprocessing: lowercasing and HTML tag removal.
    BERT tokenizer handles most other normalization.
    """
    text = text.lower()
    text = re.sub(r'<br\s*/>', ' ', text) # Remove HTML line breaks
    text = re.sub(r'<[^>]+>', '', text)   # Remove other HTML tags
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def load_imdb_data():
    """
    Loads IMDB dataset using Hugging Face datasets library.
    Returns train and test splits (texts and labels).
    """
    print("Loading IMDB dataset...")
    try:
        # Try loading with default settings
        try:
            dataset = load_dataset("imdb", trust_remote_code=True)
        except Exception as e:
            print(f"Standard load failed ({e}), trying mirror or alternative config...")
            # Fallback: try loading from a parquet export if available or simple csv if we could, 
            # but for now we just re-raise with a clear message suggesting network check.
            # In some versions, specific config verification can fail.
            raise e

    except Exception as e:
        print(f"\nCRITICAL ERROR: Could not download the IMDB dataset from Hugging Face.")
        print(f"Error details: {e}")
        print("\nPossible solutions:")
        print("1. Check your internet connection (firewall/proxy blocking Hugging Face).")
        print("2. Try running: pip install --upgrade datasets huggingface_hub")
        print("3. Validated that 'imdb' dataset is currently accessible on https://huggingface.co/datasets/imdb")
        raise e

    train_data = dataset['train']
    test_data = dataset['test']
    
    # Extract texts and labels
    train_texts = train_data['text']
    train_labels = train_data['label']
    test_texts = test_data['text']
    test_labels = test_data['label']
    
    # Subset if requested to speed up for demonstration or limited resources
    if Config.DATA_SAMPLE_SIZE:
        print(f"Subsampling dataset to {Config.DATA_SAMPLE_SIZE} records...")
        # Check if we have enough samples
        if len(train_texts) < Config.DATA_SAMPLE_SIZE:
             print(f"Warning: Training set smaller than requested sample size. Using full set ({len(train_texts)})")
        else:
             # Stratified subsample to maintain balance
             train_texts, _, train_labels, _ = train_test_split(
                 train_texts, train_labels, train_size=Config.DATA_SAMPLE_SIZE, stratify=train_labels, random_state=42
             )
        
        test_sample_size = Config.DATA_SAMPLE_SIZE // 2 if Config.DATA_SAMPLE_SIZE else None
        if len(test_texts) < test_sample_size:
             print(f"Warning: Test set smaller than requested sample size. Using full set ({len(test_texts)})")
        else:
             test_texts, _, test_labels, _ = train_test_split(
                 test_texts, test_labels, train_size=test_sample_size, stratify=test_labels, random_state=42
             )
            
    # Apply cleaning
    print("Preprocessing texts...")
    train_texts = [clean_text(t) for t in train_texts]
    test_texts = [clean_text(t) for t in test_texts]
    
    # Create validation set from training set
    train_texts, val_texts, train_labels, val_labels = train_test_split(
        train_texts, train_labels, test_size=0.1, random_state=42, stratify=train_labels
    )
    
    return (train_texts, train_labels), (val_texts, val_labels), (test_texts, test_labels)

# ==========================================
# 2. Custom Dataset Class
# ==========================================
class IMDBDataset(Dataset):
    """
    Custom PyTorch Dataset for IMDB reviews.
    """
    def __init__(self, texts, labels, tokenizer, max_len):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len
        
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, item):
        text = str(self.texts[item])
        label = self.labels[item]
        
        encoding = self.tokenizer.encode_plus(
            text,
            max_length=self.max_len,
            add_special_tokens=True,
            return_token_type_ids=False,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt',
        )
        
        return {
            'review_text': text,
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }

def create_data_loader(texts, labels, tokenizer, max_len, batch_size):
    ds = IMDBDataset(
        texts=texts,
        labels=labels,
        tokenizer=tokenizer,
        max_len=max_len
    )
    
    return DataLoader(
        ds,
        batch_size=batch_size,
        num_workers=0, # Set to 0 for Windows compatibility in some envs
        shuffle=True if batch_size is not None else False
    )

# ==========================================
# 3. Training & Evaluation Functions
# ==========================================
def train_epoch(model, data_loader, optimizer, device, scheduler, n_examples):
    model = model.train()
    losses = []
    correct_predictions = 0
    
    for batch_idx, batch in enumerate(data_loader):
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)
        
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels
        )
        
        loss = outputs.loss
        logits = outputs.logits
        
        _, preds = torch.max(logits, dim=1)
        correct_predictions += torch.sum(preds == labels)
        losses.append(loss.item())
        
        loss.backward()
        
        # Clip gradients to prevent exploding gradients
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()
        
        if batch_idx % 50 == 0:
            print(f"Batch {batch_idx}/{len(data_loader)} | Loss: {loss.item():.4f}")
            
    return correct_predictions.double() / n_examples, np.mean(losses)

def evaluate_model(model, data_loader, device, n_examples):
    model = model.eval()
    losses = []
    correct_predictions = 0
    
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for batch in data_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )
            
            loss = outputs.loss
            logits = outputs.logits
            
            _, preds = torch.max(logits, dim=1)
            correct_predictions += torch.sum(preds == labels)
            losses.append(loss.item())
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            
    accuracy = correct_predictions.double() / n_examples
    mean_loss = np.mean(losses)
    
    return accuracy, mean_loss, all_preds, all_labels

# ==========================================
# 4. Main Execution
# ==========================================
def main():
    # Load Data
    (train_texts, train_labels), (val_texts, val_labels), (test_texts, test_labels) = load_imdb_data()
    print(f"Training samples: {len(train_texts)}")
    print(f"Validation samples: {len(val_texts)}")
    print(f"Test samples: {len(test_texts)}")

    # Initialize Tokenizer
    print(f"Loading tokenizer: {Config.MODEL_NAME}")
    tokenizer = BertTokenizerFast.from_pretrained(Config.MODEL_NAME)
    
    # Create DataLoaders
    train_data_loader = create_data_loader(train_texts, train_labels, tokenizer, Config.MAX_LEN, Config.BATCH_SIZE)
    val_data_loader = create_data_loader(val_texts, val_labels, tokenizer, Config.MAX_LEN, Config.BATCH_SIZE)
    test_data_loader = create_data_loader(test_texts, test_labels, tokenizer, Config.MAX_LEN, Config.BATCH_SIZE)
    
    # Initialize Model
    print("Loading model...")
    model = BertForSequenceClassification.from_pretrained(
        Config.MODEL_NAME,
        num_labels=2, # Binary classification for Sentiment (Pos/Neg)
        output_attentions=False,
        output_hidden_states=False
    )
    model = model.to(Config.DEVICE)
    
    # Optimizer & Scheduler
    optimizer = AdamW(model.parameters(), lr=Config.LEARNING_RATE, correct_bias=False)
    total_steps = len(train_data_loader) * Config.EPOCHS
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=0,
        num_training_steps=total_steps
    )
    
    # Training Loop
    print("\nStarting Training...")
    best_accuracy = 0
    
    for epoch in range(Config.EPOCHS):
        print(f"\nEpoch {epoch + 1}/{Config.EPOCHS}")
        print('-' * 10)
        
        train_acc, train_loss = train_epoch(
            model,
            train_data_loader,
            optimizer,
            Config.DEVICE,
            scheduler,
            len(train_texts)
        )
        print(f"Train loss {train_loss:.4f} accuracy {train_acc:.4f}")
        
        val_acc, val_loss, _, _ = evaluate_model(
            model,
            val_data_loader,
            Config.DEVICE,
            len(val_texts)
        )
        print(f"Val   loss {val_loss:.4f} accuracy {val_acc:.4f}")
        
        if val_acc > best_accuracy:
            torch.save(model.state_dict(), 'best_model_state.bin')
            best_accuracy = val_acc
            
    print("\nTraining Complete.")
    
    # Final Evaluation on Test Set
    print("\nEvaluating on Test Set...")
    model.load_state_dict(torch.load('best_model_state.bin'))
    test_acc, test_loss, test_preds, test_labels = evaluate_model(
        model,
        test_data_loader,
        Config.DEVICE,
        len(test_texts)
    )
    print(f"Test Accuracy: {test_acc:.4f}")
    
    print("\nClassification Report:")
    print(classification_report(test_labels, test_preds, target_names=['Negative', 'Positive']))
    
    print("\nConfusion Matrix:")
    print(confusion_matrix(test_labels, test_preds))
    
    # Save artifacts
    print("\nSaving final model and tokenizer...")
    model.save_pretrained('./saved_model')
    tokenizer.save_pretrained('./saved_model')
    print("Model saved to ./saved_model")

def predict_sentiment(text, model_path='./saved_model'):
    """
    Inference function for a single text input.
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load model and tokenizer
    tokenizer = BertTokenizerFast.from_pretrained(model_path)
    model = BertForSequenceClassification.from_pretrained(model_path)
    model = model.to(device)
    model.eval()
    
    # Preprocess
    text = clean_text(text)
    
    # Tokenize
    encoding = tokenizer.encode_plus(
        text,
        max_length=Config.MAX_LEN,
        add_special_tokens=True,
        return_token_type_ids=False,
        padding='max_length',
        truncation=True,
        return_attention_mask=True,
        return_tensors='pt',
    )
    
    input_ids = encoding['input_ids'].to(device)
    attention_mask = encoding['attention_mask'].to(device)
    
    # Predict
    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        _, prediction = torch.max(outputs.logits, dim=1)
        
    return "Positive" if prediction.item() == 1 else "Negative"

if __name__ == "__main__":
    main()
    
    # Example Inference
    # print("\nExample Inference:")
    # print(predict_sentiment("This movie was absolutely fantastic! I loved every moment."))
