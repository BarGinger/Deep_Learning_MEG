import torch
import torch.nn as nn
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score, confusion_matrix

class MEGClassifier(nn.Module):
    """
    Deep learning model for MEG/EEG classification using a CNN-only architecture (no Transformer).
    """
    def __init__(
        self,
        tasks=["resting", "math", "memory", "motor"],
        input_channels=248,
        num_temporal_filters=8,
        temporal_kernel_size=64,
        dropout_rate=0.5,
        activation='elu',
        pool_size=4,
        num_classes=4,
        conv2d_channels=[128, 256],
        fc_hidden=128,
        fc_dropout=0.3,
        input_time_length=256
    ):
        super(MEGClassifier, self).__init__()

        if isinstance(conv2d_channels, str):
            conv2d_channels = [int(x) for x in conv2d_channels.split("_")]
        self.tasks = tasks

        activations = {
            'relu': nn.ReLU(),
            'elu': nn.ELU(),
            'gelu': nn.GELU(),
            'leakyrelu': nn.LeakyReLU(),
            'tanh': nn.Tanh()
        }
        act_fn = activations.get(activation.lower(), nn.ReLU())

        # 1D CNN for Temporal Feature Extraction
        self.cnn1d = nn.Sequential(
            nn.Conv1d(input_channels, num_temporal_filters, kernel_size=temporal_kernel_size, stride=1, padding=temporal_kernel_size // 2),
            nn.BatchNorm1d(num_temporal_filters),
            act_fn,
            nn.MaxPool1d(kernel_size=pool_size),
            nn.Dropout(dropout_rate)
        )

        # 2D CNN for Spatial Feature Extraction
        conv2d_layers = []
        in_channels = 1
        for out_channels in conv2d_channels:
            conv2d_layers.extend([
                nn.Conv2d(in_channels, out_channels, kernel_size=(3, 3), padding=(1, 1)),
                nn.BatchNorm2d(out_channels),
                act_fn,
                nn.MaxPool2d(kernel_size=(2, 2))
            ])
            in_channels = out_channels
        self.cnn2d = nn.Sequential(*conv2d_layers)

        # Calculate flattened feature size after CNNs for FC input
        with torch.no_grad():
            dummy = torch.zeros(1, input_channels, input_time_length)
            dummy = self.cnn1d(dummy)
            dummy = dummy.unsqueeze(1)
            dummy = self.cnn2d(dummy)
            batch, c, ch, t = dummy.shape
            flattened_size = c * ch * t

        # Fully Connected Classification Layer
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(flattened_size, fc_hidden),
            act_fn,
            nn.Dropout(fc_dropout),
            nn.Linear(fc_hidden, num_classes)
        )

    def forward(self, x):
        x = self.cnn1d(x)  # (batch, num_temporal_filters, time')
        x = x.unsqueeze(1) # (batch, 1, channels, time')
        x = self.cnn2d(x)  # (batch, last_conv2d_channel, ch', t')
        x = self.fc(x)     # (batch, num_classes)
        return x

    def predict(self, loader, device=None):
        preds = []
        with torch.no_grad():
            for batch in loader:
                if isinstance(batch, (list, tuple)):
                    batch_x = batch[0]
                else:
                    batch_x = batch
                if device is not None:
                    batch_x = batch_x.to(device)
                outputs = self.forward(batch_x)
                batch_preds = torch.argmax(outputs, dim=1).cpu().numpy()
                preds.extend(batch_preds)
        return np.array(preds)
        
    def evaluate(self, y_pred, y_true):
        if isinstance(y_true, torch.Tensor):
            y_true = y_true.cpu().numpy()
        if isinstance(y_pred, torch.Tensor):
            y_pred = y_pred.cpu().numpy()
        results = {}
        results["overall_accuracy"] = accuracy_score(y_true, y_pred)
        results["overall_F1"] = f1_score(y_true, y_pred, average='macro', zero_division=0)
        results["overall_recall"] = recall_score(y_true, y_pred, average='macro', zero_division=0)
        results["overall_precision"] = precision_score(y_true, y_pred, average='macro', zero_division=0)
        if self.tasks is not None and len(self.tasks) > 1:
            for idx, task in enumerate(self.tasks):
                mask = (y_true == idx)
                if np.sum(mask) == 0:
                    results[f"{task}_accuracy"] = np.nan
                    results[f"{task}_F1"] = np.nan
                    results[f"{task}_recall"] = np.nan
                    results[f"{task}_precision"] = np.nan
                else:
                    results[f"{task}_accuracy"] = accuracy_score(y_true[mask], y_pred[mask])
                    results[f"{task}_F1"] = f1_score(y_true[mask], y_pred[mask], average='macro', zero_division=0)
                    results[f"{task}_recall"] = recall_score(y_true[mask], y_pred[mask], average='macro', zero_division=0)
                    results[f"{task}_precision"] = precision_score(y_true[mask], y_pred[mask], average='macro', zero_division=0)
        return results
    

class TinyMLP(nn.Module):
    """
    Simple MLP for MEG/EEG classification. Suitable for very small datasets.
    """
    def __init__(
        self,
        tasks=["resting", "math", "memory", "motor"],
        input_channels=248,
        input_time_length=256,
        fc_hidden=32,
        fc_dropout=0.3,
        num_classes=4,
        activation='relu'
    ):
        super(TinyMLP, self).__init__()
        self.tasks = tasks

        activations = {
            'relu': nn.ReLU(),
            'elu': nn.ELU(),
            'gelu': nn.GELU(),
            'leakyrelu': nn.LeakyReLU(),
            'tanh': nn.Tanh()
        }
        act_fn = activations.get(activation.lower(), nn.ReLU())

        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(input_channels * input_time_length, fc_hidden),
            act_fn,
            nn.Dropout(fc_dropout),
            nn.Linear(fc_hidden, num_classes)
        )

    def forward(self, x):
        return self.fc(x)

    def predict(self, loader, device=None):
        preds = []
        with torch.no_grad():
            for batch in loader:
                if isinstance(batch, (list, tuple)):
                    batch_x = batch[0]
                else:
                    batch_x = batch
                if device is not None:
                    batch_x = batch_x.to(device)
                outputs = self.forward(batch_x)
                batch_preds = torch.argmax(outputs, dim=1).cpu().numpy()
                preds.extend(batch_preds)
        return np.array(preds)

    def evaluate(self, y_pred, y_true):
        if isinstance(y_true, torch.Tensor):
            y_true = y_true.cpu().numpy()
        if isinstance(y_pred, torch.Tensor):
            y_pred = y_pred.cpu().numpy()
        results = {}
        results["overall_accuracy"] = accuracy_score(y_true, y_pred)
        results["overall_F1"] = f1_score(y_true, y_pred, average='macro', zero_division=0)
        results["overall_recall"] = recall_score(y_true, y_pred, average='macro', zero_division=0)
        results["overall_precision"] = precision_score(y_true, y_pred, average='macro', zero_division=0)
        if self.tasks is not None and len(self.tasks) > 1:
            for idx, task in enumerate(self.tasks):
                mask = (y_true == idx)
                if np.sum(mask) == 0:
                    results[f"{task}_accuracy"] = np.nan
                    results[f"{task}_F1"] = np.nan
                    results[f"{task}_recall"] = np.nan
                    results[f"{task}_precision"] = np.nan
                else:
                    results[f"{task}_accuracy"] = accuracy_score(y_true[mask], y_pred[mask])
                    results[f"{task}_F1"] = f1_score(y_true[mask], y_pred[mask], average='macro', zero_division=0)
                    results[f"{task}_recall"] = recall_score(y_true[mask], y_pred[mask], average='macro', zero_division=0)
                    results[f"{task}_precision"] = precision_score(y_true[mask], y_pred[mask], average='macro', zero_division=0)
        return results