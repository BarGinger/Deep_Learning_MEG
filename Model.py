import torch
import torch.nn as nn
from torch.optim import Adam
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score, confusion_matrix
import torch.nn.functional as F

class MEGClassifier(nn.Module):
    """
    Deep learning model for MEG/EEG classification using a CNN-only architecture (no Transformer).
    """
    def __init__(
        self,
        tasks=["resting", "math", "memory", "motor"],
        number_channels=248,
        num_temporal_filters=8,
        temporal_kernel_size=64,
        dropout_rate=0.5,
        activation='elu',
        pool_size=4,
        number_classes=4,
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
            nn.Conv1d(number_channels, num_temporal_filters, kernel_size=temporal_kernel_size, stride=1, padding=temporal_kernel_size // 2),
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
            dummy = torch.zeros(1, number_channels, input_time_length)
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
            nn.Linear(fc_hidden, number_classes)
        )

    def forward(self, x):
        x = self.cnn1d(x)  # (batch, num_temporal_filters, time')
        x = x.unsqueeze(1) # (batch, 1, channels, time')
        x = self.cnn2d(x)  # (batch, last_conv2d_channel, ch', t')
        x = self.fc(x)     # (batch, number_classes)
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
    
def add_noise(data, noise_level=0.01):
    """Add Gaussian noise to MEG data."""
    noise = torch.randn_like(data) * noise_level
    return data + noise

def time_shift(data, max_shift=10):
    """Randomly shift MEG data along the time axis."""
    shift = np.random.randint(-max_shift, max_shift)
    return torch.roll(data, shifts=shift, dims=-1)

def amplitude_scale(data, scale_range=(0.9, 1.1)):
    """Randomly scale amplitude of MEG data."""
    scale = np.random.uniform(*scale_range)
    return data * scale

def channel_dropout(data, drop_prob=0.1):
    """Randomly zero out some channels in MEG data."""
    mask = (torch.rand(data.shape[-2]) > drop_prob).float()
    return data * mask[None, :, None] if data.ndim == 3 else data * mask[:, None]


class MultiviewDataset(torch.utils.data.Dataset):
    def __init__(self, X, y, window_size, mesh_rows, mesh_columns, depth, augment=False):
        """
        X: (N, C, T) torch.Tensor
        y: (N,) torch.Tensor
        window_size: int
        mesh_rows, mesh_columns, depth: int
        """
        self.X = X
        self.y = y
        self.window_size = window_size
        self.mesh_rows = mesh_rows
        self.mesh_columns = mesh_columns
        self.depth = depth
        self.augment = augment

        x_sample = self.X[0]  # shape: (C, T)
        T = x_sample.shape[1]
        split_size = T // self.window_size
        self.split_size = split_size

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        x_sample = self.X[idx]  # shape: (C, T)

        if self.augment:
            # Randomly choose which augmentations to apply
            if np.random.rand() < 0.7:
                x_sample = add_noise(x_sample, noise_level=0.05)
            if np.random.rand() < 0.7:
                x_sample = time_shift(x_sample, max_shift=20)
            if np.random.rand() < 0.5:
                x_sample = amplitude_scale(x_sample, scale_range=(0.8, 1.2))
            if np.random.rand() < 0.5:
                x_sample = channel_dropout(x_sample, drop_prob=0.2)

        T = x_sample.shape[1]
        split_size = T // self.window_size
        if split_size != self.mesh_rows * self.mesh_columns:
            raise ValueError(
                f"split_size ({split_size}) != mesh_rows * mesh_columns ({self.mesh_rows * self.mesh_columns}). "
                f"Check your window_size, mesh_rows, mesh_columns, and input data shape."
            )
        x_cnn_list = []
        x_lstm_list = []
        for i in range(self.window_size):
            start = i * split_size
            end = start + split_size
            window = x_sample[:, start:end]  # shape: (C, split_size)
            x_cnn = window[:self.depth].reshape(self.depth, self.mesh_rows, self.mesh_columns)
            x_cnn_list.append(x_cnn)
            x_lstm = window[:self.depth].T  # shape: (split_size, depth)
            x_lstm_list.append(x_lstm)
        return x_cnn_list, x_lstm_list, self.y[idx]    
    
    
# Add this helper class at the top of your Model.py
class Permute(nn.Module):
    def __init__(self, *dims):
        super().__init__()
        self.dims = dims
    def forward(self, x):
        return x.permute(*self.dims)

class MultiviewAttention(nn.Module):
    def __init__(
        self,
        window_size,
        conv1_filters,
        conv2_filters,
        conv3_filters,
        conv1_kernel_shape,
        conv2_kernel_shape,
        conv3_kernel_shape,
        padding1,
        padding2,
        padding3,
        conv1_activation,
        conv2_activation,
        conv3_activation,
        dense_nodes,
        dense_activation,
        depth,
        lstm1_cells,
        lstm2_cells,
        dense3_nodes,
        dense3_activation,
        dropout_rate=0.3,  # Suggest lower default dropout
        number_classes=4,
        number_channels=248,
        mesh_rows=20,
        mesh_columns=21,
        split_size=None,
        transformer_nhead=4,
        transformer_num_layers=1

    ):
        super().__init__()
        self.window_size = window_size
        self.number_classes = number_classes
        self.number_channels = number_channels
        self.mesh_rows = mesh_rows
        self.mesh_columns = mesh_columns
        self.depth = depth
        self.tasks = ["resting", "math", "memory", "motor"]
        self.split_size = split_size
        self.dropout_rate = dropout_rate
        self.dropout = nn.Dropout(self.dropout_rate)
        self.transformer_nhead = transformer_nhead
        self.transformer_num_layers = transformer_num_layers

        # Activation functions
        activations = {
            'relu': nn.ReLU(),
            'elu': nn.ELU(),
            'gelu': nn.GELU(),
            'leakyrelu': nn.LeakyReLU(),
            'tanh': nn.Tanh()
        }
        act1 = activations.get(conv1_activation.lower(), nn.ReLU())
        act2 = activations.get(conv2_activation.lower(), nn.ReLU())
        act3 = activations.get(conv3_activation.lower(), nn.ReLU())
        act_dense = activations.get(dense_activation.lower(), nn.ReLU())
        act_dense3 = activations.get(dense3_activation.lower(), nn.ReLU())

        # CNN branches for each window (up to flatten)
        self.cnn_branches = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(depth, conv1_filters, kernel_size=conv1_kernel_shape, padding=padding1),
                nn.BatchNorm2d(conv1_filters),
                act1,
                nn.Dropout(self.dropout_rate),
                
                nn.Conv2d(conv1_filters, conv2_filters, kernel_size=conv2_kernel_shape, padding=padding2),
                nn.BatchNorm2d(conv2_filters),
                act2,
                # Removed dropout here for less regularization
                
                nn.Conv2d(conv2_filters, conv3_filters, kernel_size=conv3_kernel_shape, padding=padding3),
                nn.BatchNorm2d(conv3_filters),
                act3,
                nn.AdaptiveAvgPool2d((1, 1)),
                nn.Flatten(start_dim=1)
            ) for _ in range(window_size)
        ])

        # Dynamically compute flattened size for FC layer
        with torch.no_grad():
            dummy = torch.zeros(1, depth, mesh_rows, mesh_columns)
            dummy_out = self.cnn_branches[0](dummy)
            flattened_size = dummy_out.shape[1]

        # Dense after CNN branch
        self.cnn_dense = nn.ModuleList([
            nn.Sequential(
                nn.Linear(flattened_size, dense_nodes),
                act_dense,
                nn.Dropout(self.dropout_rate) 
            ) for _ in range(window_size)
        ])

        # Self-attention mechanism
        self.attention = nn.MultiheadAttention(embed_dim=dense_nodes, num_heads=transformer_nhead, batch_first=True)

        # LSTM branches for each window
        self.lstm_dense = nn.ModuleList([
            nn.Sequential(
                nn.Linear(depth, dense_nodes),
                act_dense
            ) for _ in range(window_size)
        ])

        self.lstm_temporal_attention = nn.Sequential(
            nn.Linear(dense_nodes, 64),
            nn.Tanh(),
            nn.Linear(64, 1)
        )

        # LSTM layers
        self.lstm1 = nn.LSTM(dense_nodes, lstm1_cells, batch_first=True)
        self.lstm2 = nn.LSTM(lstm1_cells, lstm2_cells, batch_first=True)

        # Transformer encoder block (NEW)
        self.transformer_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=dense_nodes, nhead=transformer_nhead, batch_first=True, dropout=self.dropout_rate),
            num_layers=transformer_num_layers
        )

        # Dense after LSTM
        self.dense3 = nn.Linear(lstm2_cells, dense3_nodes)
        self.dense3_act = act_dense3

        # Final layers
        self.final_add = nn.Linear(dense_nodes, dense_nodes)  # for addition, if needed
        # +dense_nodes for transformer output
        self.final_out = nn.Linear(dense3_nodes + dense_nodes + dense_nodes + dense_nodes, number_classes)

    def forward(self, x_cnn_list, x_lstm_list):
        cnn_outs = []
        for i, (branch, dense) in enumerate(zip(self.cnn_branches, self.cnn_dense)):
            out = branch(x_cnn_list[i])
            out = dense(out)
            cnn_outs.append(out)
        
        lstm_in = []
        for branch, x_lstm in zip(self.lstm_dense, x_lstm_list):
            out = branch(x_lstm)  # (batch, split_size, dense_nodes)
            attn_weights = self.lstm_temporal_attention(out).squeeze(-1)  # (batch, split_size)
            attn_weights = F.softmax(attn_weights, dim=1).unsqueeze(-1)   # (batch, split_size, 1)
            out = (out * attn_weights).sum(dim=1)  # Weighted sum: (batch, dense_nodes)
            lstm_in.append(out)
        lstm_in = torch.stack(lstm_in, dim=1)  # (batch, window_size, dense_nodes)
        
        # Self-attention across windows
        attn_output, _ = self.attention(lstm_in, lstm_in, lstm_in)
        attn_feat = self.dropout(attn_output.mean(dim=1))

        # Transformer encoder (NEW)
        transformer_out = self.transformer_encoder(lstm_in)
        transformer_feat = self.dropout(transformer_out.mean(dim=1))

        # LSTM
        lstm_out, _ = self.lstm1(lstm_in)
        lstm_out, _ = self.lstm2(lstm_out)
        lstm_feat = self.dense3_act(self.dense3(lstm_out[:, -1, :]))
        lstm_feat = self.dropout(lstm_feat)

        added = self.dropout(torch.stack(cnn_outs, dim=0).sum(dim=0))
        final = torch.cat([lstm_feat, added, attn_feat, transformer_feat], dim=-1)
        final = self.dropout(final)

        out = self.final_out(final)
        return out

    def predict(self, loader, device=None):
        preds = []
        self.eval()
        with torch.no_grad():
            for batch in loader:
                if isinstance(batch, (list, tuple)):
                    batch_x_cnn, batch_x_lstm = batch[0], batch[1]
                else:
                    batch_x_cnn, batch_x_lstm = batch, batch
                if device is not None:
                    batch_x_cnn = [x.to(device) for x in batch_x_cnn]
                    batch_x_lstm = [x.to(device) for x in batch_x_lstm]
                outputs = self.forward(batch_x_cnn, batch_x_lstm)
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
        if hasattr(self, "tasks") and self.tasks is not None and len(self.tasks) > 1:
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

class EEGNet(nn.Module):
    def __init__(self, nb_classes, Channels, Samples, conv1_filters, conv2_filters,
                 dropoutRate=0.5, kernLength=64,  
                 depth=2, norm_rate=0.25, dropoutType='Dropout'):
        super(EEGNet, self).__init__()

        if dropoutType == 'SpatialDropout2D':
            self.dropout = nn.Dropout2d
        elif dropoutType == 'Dropout':
            self.dropout = nn.Dropout
        else:
            raise ValueError("dropoutType must be one of 'SpatialDropout2D' or 'Dropout'.")

        self.firstconv = nn.Sequential(
            nn.Conv2d(1, conv1_filters, kernel_size=(1, kernLength), padding='same', bias=False),
            nn.BatchNorm2d(conv1_filters)
        )
        

        self.depthwiseConv = nn.Sequential(
            nn.Conv2d(conv1_filters, conv1_filters * depth, kernel_size=(Channels, 1), groups=conv1_filters, bias=False),
            nn.BatchNorm2d(conv1_filters * depth),
            nn.ELU(),
            nn.AvgPool2d(kernel_size=(1, 4)),
            self.dropout(dropoutRate)
        )

        self.separableConv = nn.Sequential(
            nn.Conv2d(conv1_filters * depth, conv1_filters * depth, kernel_size=(1, 16), groups=conv1_filters * depth, padding='same', bias=False),
            nn.Conv2d(conv1_filters * depth, conv2_filters, kernel_size=1, bias=False),
            nn.BatchNorm2d(conv2_filters),
            nn.ELU(),
            nn.AvgPool2d(kernel_size=(1, 8)),
            self.dropout(dropoutRate)
        )

        # Compute output size after conv/pool layers
        self._output_size = self._get_flattened_size(Channels, Samples)

        self.classifier = nn.Sequential(
            nn.Linear(self._output_size, nb_classes),
            nn.Softmax(dim=1)
        )
      

    def _get_flattened_size(self, Channels, Samples):
        with torch.no_grad():
            x = torch.zeros(1, 1, Channels, Samples)
            x = self.firstconv(x)
            x = self.depthwiseConv(x)
            x = self.separableConv(x)
            return x.view(1, -1).shape[1]

    def forward(self, x):
        x = self.firstconv(x)
        x = self.depthwiseConv(x)
        x = self.separableConv(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x


import torch
import torch.nn as nn
import torch.nn.functional as F

class SimpleAttention(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.attn = nn.Linear(input_dim, 1)

    def forward(self, x):
        # x: (batch, seq_len, input_dim)
        weights = F.softmax(self.attn(x), dim=1)  # (batch, seq_len, 1)
        return (x * weights).sum(dim=1)  # (batch, input_dim)

class MultiviewAttentionRef(nn.Module):
    def __init__(self, window_size, depth, mesh_rows, mesh_columns, cnn_filters, lstm_hidden, dense_hidden, num_classes, dropout_rate=0.1):
        super().__init__()
        self.window_size = window_size
        self.depth = depth
        self.mesh_rows = mesh_rows
        self.mesh_columns = mesh_columns
        self.cnn_filters = cnn_filters
        self.lstm_hidden = lstm_hidden
        self.dense_hidden = dense_hidden
        self.num_classes = num_classes
        self.dropout_rate = dropout_rate

        # Shared CNN for all windows
        self.cnn_branch = nn.Sequential(
            nn.Conv2d(depth, cnn_filters, kernel_size=3, padding=1),
            nn.BatchNorm2d(cnn_filters),
            nn.ELU(),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten()
        )

        # Shared Dense + Dropout
        self.dense = nn.Sequential(
            nn.Linear(cnn_filters, dense_hidden),
            nn.ELU(),
            nn.Dropout(dropout_rate)
        )

        # Shared LSTM + Attention
        self.lstm = nn.LSTM(dense_hidden, lstm_hidden, batch_first=True, dropout=self.dropout_rate, num_layers=2)
        # Attention mechanism
        self.attn = SimpleAttention(lstm_hidden)

        # Final classifier
        self.final_dense = nn.Sequential(
            nn.Linear(lstm_hidden, 64),
            nn.ELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(64, num_classes)
        )

    def forward(self, x_cnn_list):
        batch_size = x_cnn_list[0].size(0)
        cnn_feats = []

        for x in x_cnn_list:
            x = x.reshape(batch_size, self.depth, self.mesh_rows, self.mesh_columns)
            x = self.cnn_branch(x)  # (batch, cnn_filters)
            x = self.dense(x)       # (batch, dense_hidden)
            cnn_feats.append(x)

        cnn_seq = torch.stack(cnn_feats, dim=1)  # (batch, window_size, dense_hidden)
        lstm_out, _ = self.lstm(cnn_seq)         # (batch, window_size, lstm_hidden)
        attn_out = self.attn(lstm_out)           # (batch, lstm_hidden)
        attn_out = F.dropout(attn_out, p=self.dropout_rate, training=self.training)  # ✅ Additional regularization
        return self.final_dense(attn_out)        # (batch, num_classes)
    
    def predict(self, loader, device):
        self.eval()
        preds = []
        with torch.no_grad():
            for batch in loader:
                batch_x_cnn, batch_x_lstm, _ = batch  # unpack all, but only use batch_x_cnn
                batch_x_cnn = [x.to(device) for x in batch_x_cnn]
                outputs = self.forward(batch_x_cnn)  # Only pass batch_x_cnn!
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
        if hasattr(self, "tasks") and self.tasks is not None and len(self.tasks) > 1:
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