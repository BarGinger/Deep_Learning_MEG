import torch
import torch.nn as nn
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score

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

    def predict(self, x, device=None, batch_size=64):
        """
        Predict class indices for input x.

        Parameters
        ----------
        x : np.ndarray or torch.Tensor
            Input data of shape (n_samples, channels, time).
        device : torch.device or None
            Device to run prediction on.
        batch_size : int
            Batch size for prediction.

        Returns
        -------
        np.ndarray
            Predicted class indices.
        """
        self.eval()
        if isinstance(x, np.ndarray):
            x = torch.tensor(x, dtype=torch.float32)
        if device is not None:
            x = x.to(device)
        preds = []
        with torch.no_grad():
            for i in range(0, len(x), batch_size):
                batch_x = x[i:i+batch_size]
                outputs = self.forward(batch_x)
                batch_preds = torch.argmax(outputs, dim=1).cpu().numpy()
                preds.extend(batch_preds)
        return np.array(preds)

    def evaluate(self, x, y_true, device=None, batch_size=64, compute_metrics_fn=None):
        """
        Evaluate the model on data x with true labels y_true.

        Parameters
        ----------
        x : np.ndarray or torch.Tensor
            Input data of shape (n_samples, channels, time).
        y_true : np.ndarray or list
            True class indices.
        device : torch.device or None
            Device to run evaluation on.
        batch_size : int
            Batch size for evaluation.
        compute_metrics_fn : callable or None
            Function to compute metrics (e.g., utils.compute_metrics). If None, uses basic sklearn metrics.

        Returns
        -------
        dict
            Dictionary with accuracy, F1, recall, and precision.
        """
        y_pred = self.predict(x, device=device, batch_size=batch_size)
        if compute_metrics_fn is not None:
            metrics = compute_metrics_fn(y_true, y_pred)
            return metrics
        else:
            return {
                "accuracy": accuracy_score(y_true, y_pred),
                "F1": f1_score(y_true, y_pred, average='macro', zero_division=0),
                "recall": recall_score(y_true, y_pred, average='macro', zero_division=0),
                "precision": precision_score(y_true, y_pred, average='macro', zero_division=0)
            }