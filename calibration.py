
import torch
import torch.nn as nn
import torch.optim as optim

class ModelCalibrator:
    """
    Implements Temperature Scaling to calibrate model probabilities.
    Ref: On Calibration of Modern Neural Networks (Guo et al. 2017)
    """
    def __init__(self, model, device):
        self.model = model
        self.device = device
        self.temperature = nn.Parameter(torch.ones(1) * 1.5) # Initialize T > 1
        self.temperature.to(device)

    def train_calibration(self, val_loader):
        """
        Tune the temperature parameter using validation set.
        """
        self.model.eval()
        nll_criterion = nn.CrossEntropyLoss()
        optimizer = optim.LBFGS([self.temperature], lr=0.01, max_iter=50)

        # Collect logits and labels
        logits_list = []
        labels_list = []
        
        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['labels'].to(self.device)
                
                outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
                logits_list.append(outputs.logits)
                labels_list.append(labels)
                
        logits = torch.cat(logits_list).to(self.device)
        labels = torch.cat(labels_list).to(self.device)

        def eval():
            optimizer.zero_grad()
            # Scale logits by temperature
            loss = nll_criterion(logits / self.temperature, labels)
            loss.backward()
            return loss

        optimizer.step(eval)
        
        print(f"Optimal Temperature: {self.temperature.item():.4f}")
        return self.temperature.item()

    def calibrate_logits(self, logits):
        """
        Scales logits using the learned temperature.
        """
        return logits / self.temperature
