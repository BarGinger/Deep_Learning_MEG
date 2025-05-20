import torch
import torch.nn as nn
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score, confusion_matrix

class MEGClassifier(nn.Module):
    """
    Deep learning model for MEG/EEG classification using a hybrid CNN + Transformer architecture.

    Parameters
    ----------
    input_channels : int, default=248
        Number of input channels (sensors).
    num_temporal_filters : int
        Number of temporal filters in the 1D CNN.
    temporal_kernel_size : int
        Kernel size for the 1D CNN.
    dropout_rate : float
        Dropout rate after 1D CNN.
    activation : str
        Activation function to use ('relu', 'elu', etc.).
    pool_size : int
        Pooling size for 1D CNN.
    num_classes : int, default=4
        Number of output classes.
    conv2d_channels : list of int
        List specifying the number of channels for each 2D CNN layer.
    transformer_d_model : int
        Model dimension for the transformer encoder.
    transformer_nhead : int
        Number of attention heads in the transformer encoder.
    transformer_dim_feedforward : int
        Feedforward dimension in the transformer encoder.
    transformer_layers : int
        Number of transformer encoder layers.
    fc_hidden : int
        Number of hidden units in the fully connected layer.
    fc_dropout : float
        Dropout rate in the fully connected layer.
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
        transformer_d_model=256,
        transformer_nhead=8,
        transformer_dim_feedforward=512,
        transformer_layers=4,
        fc_hidden=128,
        fc_dropout=0.3
    ):
        super(MEGClassifier, self).__init__()

        # format conv2d_channels if it's a string to a list
        if isinstance(conv2d_channels, str):
            conv2d_channels = [int(x) for x in conv2d_channels.split("_")]
        
        # save tasks
        self.tasks = tasks

        # Activation function selection
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
        prev_channels = num_temporal_filters
        for out_channels in conv2d_channels:
            conv2d_layers.extend([
                nn.Conv2d(in_channels, out_channels, kernel_size=(3, 3), padding=(1, 1)),
                nn.BatchNorm2d(out_channels),
                act_fn,
                nn.MaxPool2d(kernel_size=(2, 2))
            ])
            in_channels = out_channels
        self.cnn2d = nn.Sequential(*conv2d_layers)

        # Calculate flattened feature size after CNNs for transformer input
        # We'll assume input shape (batch, channels, time)
        # After 1D CNN: (batch, num_temporal_filters, time//pool_size)
        # After 2D CNN: (batch, last_conv2d_channel, reduced_ch, reduced_time)
        # We'll use transformer_d_model as the feature size for the transformer
        self.transformer_d_model = transformer_d_model

        with torch.no_grad():
            dummy = torch.zeros(1, input_channels, 512)  # 512 is an example time length
            dummy = self.cnn1d(dummy)
            dummy = dummy.unsqueeze(1)
            dummy = self.cnn2d(dummy)
            batch, c, ch, t = dummy.shape
            flattened_size = ch * c

        # Linear layer to project CNN2D output to transformer_d_model if needed
        self.project_to_transformer = nn.Linear(flattened_size, transformer_d_model)
        # self.project_to_transformer = nn.Linear(conv2d_channels[-1], transformer_d_model)

        # Transformer Encoder for Temporal Attention
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=transformer_d_model,
                nhead=transformer_nhead,
                dim_feedforward=transformer_dim_feedforward,
                activation=activation,
                batch_first=True
            ),
            num_layers=transformer_layers
        )

        # Fully Connected Classification Layer
        self.fc = nn.Sequential(
            nn.Linear(transformer_d_model, fc_hidden),
            act_fn,
            nn.Dropout(fc_dropout),
            nn.Linear(fc_hidden, num_classes)
        )

    def forward(self, x):
        """
        Forward pass of the model.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (batch_size, channels, time).

        Returns
        -------
        torch.Tensor
            Output logits of shape (batch_size, num_classes).
        """
        x = self.cnn1d(x)  # (batch, num_temporal_filters, time')
        # Reshape for 2D CNN: (batch, 1, channels, time')
        x = x.unsqueeze(1)
        x = self.cnn2d(x)  # (batch, last_conv2d_channel, ch', t')
        batch, c, ch, t = x.shape
        # Flatten spatial dims, keep time as sequence: (batch, t, c * ch)
        x = x.permute(0, 3, 1, 2).contiguous().view(batch, t, c * ch)
        # Project to transformer_d_model if needed
        x = self.project_to_transformer(x)  # (batch, t, transformer_d_model)
        # Pass through transformer (batch_first=True)
        x = self.transformer(x)  # (batch, t, transformer_d_model)
        # Global Average Pooling over sequence
        x = x.mean(dim=1)  # (batch, transformer_d_model)
        x = self.fc(x)     # (batch, num_classes)
        return x

    def predict(self, loader, device=None):
        """
        Predict class indices for input x.

        Parameters
        ----------
        loader : torch.utils.data.DataLoader
            DataLoader for the input data.
        device : torch.device or None
            Device to run prediction on.

        Returns
        -------
        np.ndarray
            Predicted class indices.
        """
        preds = []
        with torch.no_grad():
            for batch in loader:
                # Unpack batch if it's a tuple/list (inputs, labels)
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
        """
        Evaluate the model on data x with true labels y_true.

        Parameters
        ----------
        y_pred : np.ndarray or list or torch.Tensor
            Predicted class indices.
        y_true : np.ndarray or list or torch.Tensor
            True class indices.

        Returns
        -------
        dict
            Dictionary with accuracy, F1, recall, and precision.
        """
        # Ensure numpy arrays for compatibility with sklearn and numpy
        if isinstance(y_true, torch.Tensor):
            y_true = y_true.cpu().numpy()

        if isinstance(y_pred, torch.Tensor):
            y_pred = y_pred.cpu().numpy()
        
        # print(confusion_matrix(y_true, y_pred))

        results = {}
        # Overall
        results["overall_accuracy"] = accuracy_score(y_true, y_pred)
        results["overall_F1"] = f1_score(y_true, y_pred, average='macro', zero_division=0)
        results["overall_recall"] = recall_score(y_true, y_pred, average='macro', zero_division=0)
        results["overall_precision"] = precision_score(y_true, y_pred, average='macro', zero_division=0)

        # Per task
        if self.tasks is not None and len(self.tasks) > 1:
            for idx, task in enumerate(self.tasks):
                mask = (y_true == idx)
                if np.sum(mask) == 0:
                    # No samples for this task in this split
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