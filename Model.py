import torch
import torch.nn as nn
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score, confusion_matrix
import torch.nn.functional as F
from torchsummary import summary #just for debugging


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
    def __init__(self, X, y, window_size, mesh_rows, mesh_columns, depth, augment=False, summary_fn='mean'):
        """
        X: (N, C, T) torch.Tensor
        y: (N,) torch.Tensor
        window_size: int
        mesh_rows, mesh_columns, depth: int
        summary_fn: 'mean' or 'max' or 'first' (how to summarize time window for CNN)
        """
        self.X = X
        self.y = y
        self.window_size = window_size
        self.mesh_rows = mesh_rows
        self.mesh_columns = mesh_columns
        self.depth = depth
        self.augment = augment
        self.summary_fn = summary_fn

        x_sample = self.X[0]  # shape: (C, T)
        T = x_sample.shape[1]
        self.split_size = T // self.window_size
        

        # Add this check:
        number_channels = x_sample.shape[0]
        if self.depth * self.mesh_rows * self.mesh_columns != number_channels:
            raise ValueError(
                f"depth*mesh_rows*mesh_columns ({self.depth}*{self.mesh_rows}*{self.mesh_columns}) "
                f"must equal number_channels ({number_channels})"
            )

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        x_sample = self.X[idx]  # shape: (number_channels, T)

        if self.augment:
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

        x_cnn_list = []
        x_lstm_list = []
        for i in range(self.window_size):
            start = i * split_size
            end = start + split_size
            window = x_sample[:, start:end]  # (number_channels, split_size)

            # CNN branch: summarize over time, then reshape to (depth, mesh_rows, mesh_columns)
            if self.summary_fn == 'mean':
                window_summary = window.mean(axis=1)  # (number_channels,)
            elif self.summary_fn == 'max':
                window_summary = window.max(axis=1)
            elif self.summary_fn == 'first':
                window_summary = window[:, 0]
            else:
                raise ValueError("Unknown summary_fn")

            # Reshape to (depth, mesh_rows, mesh_columns)
            x_cnn = window_summary.reshape(self.depth, self.mesh_rows, self.mesh_columns)

            x_cnn_list.append(x_cnn)

            # LSTM branch: (split_size, number_channels)
            x_lstm = window.T  # (split_size, number_channels)
            x_lstm_list.append(x_lstm)

        return x_cnn_list, x_lstm_list, self.y[idx]

class SelfAttention2D(nn.Module):
    def __init__(self, in_channels, dk, dv, n_heads, relative=False):
        super().__init__()
        self.dk = dk
        self.dv = dv
        self.n_heads = n_heads
        self.relative = relative

        self.qkv_conv = nn.Conv2d(in_channels, 2 * dk + dv, kernel_size=1)
        self.output_conv = nn.Conv2d(dv, dv, kernel_size=1)

        # --- Add relative positional embeddings ---
        if self.relative:
            # These will be initialized later when input size is known
            self.rel_height = None
            self.rel_width = None

    def _get_rel_embeddings(self, H, W, device):
        # Only create if not already created or shape mismatch
        if (self.rel_height is None) or (self.rel_height.shape[0] != 2*H-1):
            self.rel_height = nn.Parameter(torch.randn(2*H-1, self.dk // self.n_heads, device=device) * (self.dk // self.n_heads) ** -0.5)
        if (self.rel_width is None) or (self.rel_width.shape[0] != 2*W-1):
            self.rel_width = nn.Parameter(torch.randn(2*W-1, self.dk // self.n_heads, device=device) * (self.dk // self.n_heads) ** -0.5)

    def _relative_logits_1d(self, q, rel_k, H, W, transpose_mask):
        # q: (B, n_heads, H, W, d_head)
        # rel_k: (2*W-1, d_head)
        rel_logits = torch.einsum('bhlwd,md->bhlwm', q, rel_k)  # (B, n_heads, H, W, 2W-1)
        rel_logits = self._rel_to_abs(rel_logits)
        rel_logits = rel_logits.unsqueeze(3).expand(-1, -1, -1, H, -1, -1)  # (B, n_heads, H, H, W, W)
        rel_logits = rel_logits.permute(*transpose_mask)
        rel_logits = rel_logits.reshape(q.shape[0], self.n_heads, H*W, H*W)
        return rel_logits
    
    def _rel_to_abs(self, x):
        # x: (B, n_heads, L, W, 2W-1)
        B, n_heads, L, W, _ = x.shape
        col_pad = torch.zeros((B, n_heads, L, W, 1), device=x.device, dtype=x.dtype)
        x = torch.cat([x, col_pad], dim=-1)
        flat_x = x.view(B, n_heads, L, -1)
        flat_pad = torch.zeros((B, n_heads, L, W-1), device=x.device, dtype=x.dtype)
        flat_x_padded = torch.cat([flat_x, flat_pad], dim=-1)
        final_x = flat_x_padded.view(B, n_heads, L, W+1, 2*W-1)
        final_x = final_x[:, :, :, :W, W-1:]
        return final_x
    
    def forward(self, x):
        B, C, H, W = x.shape
        qkv = self.qkv_conv(x)
        q, k, v = torch.split(qkv, [self.dk, self.dk, self.dv], dim=1)

        def reshape_heads(tensor, head_dim):
            return tensor.view(B, self.n_heads, head_dim // self.n_heads, H, W).permute(0, 1, 3, 4, 2)  # (B, n_heads, H, W, d_head)

        q = reshape_heads(q, self.dk)
        k = reshape_heads(k, self.dk)
        v = reshape_heads(v, self.dv)

        # Flatten spatial dims for dot-product attention
        q_flat = q.reshape(B, self.n_heads, H*W, -1)
        k_flat = k.reshape(B, self.n_heads, H*W, -1)
        v_flat = v.reshape(B, self.n_heads, H*W, -1)

        attn_logits = torch.matmul(q_flat, k_flat.transpose(-2, -1))  # (B, n_heads, HW, HW)
        attn_logits = attn_logits / (self.dk // self.n_heads) ** 0.5

        # --- Add relative positional logits if enabled ---
        if self.relative:
            self._get_rel_embeddings(H, W, x.device)
            # Height
            rel_logits_h = self._relative_logits_1d(q, self.rel_height, H, W, [0, 1, 4, 2, 5, 3])
            # Width
            rel_logits_w = self._relative_logits_1d(q.permute(0, 1, 4, 3, 2), self.rel_width, W, H, [0, 1, 3, 4, 2, 5])
            attn_logits = attn_logits + rel_logits_h + rel_logits_w

        attn_weights = F.softmax(attn_logits, dim=-1)
        attn_out = torch.matmul(attn_weights, v_flat)  # (B, n_heads, HW, d_v_head)
        attn_out = attn_out.view(B, self.n_heads, H, W, self.dv // self.n_heads).permute(0, 1, 4, 2, 3).contiguous()
        attn_out = attn_out.view(B, self.dv, H, W)
        out = self.output_conv(attn_out)  # (B, dv, H, W)
        return out


class AugmentedConv2D(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, dk, dv, n_heads, relative):
        super().__init__()
        conv_channels = max(out_channels - dv, 1)
        self.conv = nn.Conv2d(in_channels, conv_channels, kernel_size=kernel_size, padding='same')
        self.attn = SelfAttention2D(in_channels, dk, dv, n_heads, relative)

    def forward(self, x):
        return torch.cat([self.conv(x), self.attn(x)], dim=1)


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
        number_channels,
        number_of_heads,
        mesh_rows,
        mesh_columns,
        number_classes,
    ):
        super().__init__()
        self.window_size = window_size
        self.depth = depth
        self.channels = number_channels
        self.mesh_rows = mesh_rows
        self.mesh_columns = mesh_columns
        self.number_classes = number_classes

        activations = {
            'relu': nn.ReLU(),
            'elu': nn.ELU(),
            'gelu': nn.GELU(),
            'leakyrelu': nn.LeakyReLU(),
            'tanh': nn.Tanh()
        }
        self.act1 = activations.get(conv1_activation.lower(), nn.ReLU())
        self.act2 = activations.get(conv2_activation.lower(), nn.ReLU())
        self.act3 = activations.get(conv3_activation.lower(), nn.ReLU())
        self.act_dense = activations.get(dense_activation.lower(), nn.ReLU())
        self.act_dense3 = activations.get(dense3_activation.lower(), nn.ReLU())

        # CNN branches
        self.cnn_branches = nn.ModuleList()
        self.cnn_dense = nn.ModuleList()
        for _ in range(window_size):
            branch = nn.Sequential(
                AugmentedConv2D(depth, conv1_filters, conv1_kernel_shape, dk=6, dv=4, n_heads=2, relative=True),
                nn.BatchNorm2d(conv1_filters),
                self.act1,
                nn.Conv2d(conv1_filters, conv2_filters, kernel_size=conv2_kernel_shape, padding=padding2),
                self.act2,
                nn.Conv2d(conv2_filters, conv3_filters, kernel_size=conv3_kernel_shape, padding=padding3),
                self.act3,
                nn.Flatten()
            )
            self.cnn_branches.append(branch)
            # Compute flat size dynamically
            with torch.no_grad():
                dummy = torch.zeros(1, depth, mesh_rows, mesh_columns)
                flat_size = branch(dummy).shape[1]
            self.cnn_dense.append(
                nn.Sequential(
                    nn.Linear(flat_size, dense_nodes),
                    self.act_dense
                )
            )

        # LSTM branches
        self.lstm_dense = nn.ModuleList([
            nn.Sequential(
                nn.Linear(number_channels, dense_nodes),
                self.act_dense
            ) for _ in range(window_size)
        ])
        self.lstm1 = nn.LSTM(dense_nodes, lstm1_cells, batch_first=True)
        self.lstm2 = nn.LSTM(lstm1_cells, lstm2_cells, batch_first=True)
        self.final_dense = nn.Sequential(
            nn.Linear(lstm2_cells, dense3_nodes),
            self.act_dense3
        )

        # Final classifier
        self.classifier = nn.Linear(dense3_nodes + dense_nodes, number_classes)
        self.tasks = ["resting", "math", "memory", "motor"]

    def forward(self, x_cnn_list, x_lstm_list):
        # x_cnn_list: list of [batch, depth, mesh_rows, mesh_columns]
        # x_lstm_list: list of [batch, number_channels, depth]
        cnn_outputs = []
        for i in range(self.window_size):
            x = self.cnn_branches[i](x_cnn_list[i])
            x = self.cnn_dense[i](x)
            cnn_outputs.append(x)
        cnn_sum = torch.stack(cnn_outputs, dim=0).sum(dim=0)  # (batch, dense_nodes)

        # LSTM branch
        lstm_inputs = []
        for i in range(self.window_size):
            # Permute to (batch, depth, number_channels) to match Keras Permute(2,1)
            # x = x_lstm_list[i].permute(0, 2, 1)
            x = x_lstm_list[i]
            x = self.lstm_dense[i](x)
            lstm_inputs.append(x)
        # Stack along window axis: (batch, window_size, depth, dense_nodes)
        lstm_inputs = torch.stack(lstm_inputs, dim=1)  # (batch, window_size, depth, dense_nodes)
        # Merge depth and window_size: (batch, window_size * depth, dense_nodes)
        batch, window, depth, dnodes = lstm_inputs.shape
        lstm_inputs = lstm_inputs.view(batch, window * depth, dnodes)
        # LSTM layers
        x, _ = self.lstm1(lstm_inputs)
        x, _ = self.lstm2(x)
        x = self.final_dense(x[:, -1, :])  # Take last time step

        # Final concat and classifier
        out = torch.cat([x, cnn_sum], dim=1)
        return self.classifier(out)

    def predict(self, loader, device):
        self.eval()
        preds = []
        with torch.no_grad():
            for batch in loader:
                batch_x_cnn, batch_x_lstm, _ = batch
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



# class MEGClassifier(nn.Module):
#     """
#     Deep learning model for MEG/EEG classification using a CNN-only architecture (no Transformer).
#     """
#     def __init__(
#         self,
#         tasks=["resting", "math", "memory", "motor"],
#         number_channels=248,
#         num_temporal_filters=8,
#         temporal_kernel_size=64,
#         dropout_rate=0.5,
#         activation='elu',
#         pool_size=4,
#         number_classes=4,
#         conv2d_channels=[128, 256],
#         fc_hidden=128,
#         fc_dropout=0.3,
#         input_time_length=256
#     ):
#         super(MEGClassifier, self).__init__()

#         if isinstance(conv2d_channels, str):
#             conv2d_channels = [int(x) for x in conv2d_channels.split("_")]
#         self.tasks = tasks

        # activations = {
        #     'relu': nn.ReLU(),
        #     'elu': nn.ELU(),
        #     'gelu': nn.GELU(),
        #     'leakyrelu': nn.LeakyReLU(),
        #     'tanh': nn.Tanh()
        # }
#         act_fn = activations.get(activation.lower(), nn.ReLU())

#         # 1D CNN for Temporal Feature Extraction
#         self.cnn1d = nn.Sequential(
#             nn.Conv1d(number_channels, num_temporal_filters, kernel_size=temporal_kernel_size, stride=1, padding=temporal_kernel_size // 2),
#             nn.BatchNorm1d(num_temporal_filters),
#             act_fn,
#             nn.MaxPool1d(kernel_size=pool_size),
#             nn.Dropout(dropout_rate)
#         )

#         # 2D CNN for Spatial Feature Extraction
#         conv2d_layers = []
#         in_channels = 1
#         for out_channels in conv2d_channels:
#             conv2d_layers.extend([
#                 nn.Conv2d(in_channels, out_channels, kernel_size=(3, 3), padding=(1, 1)),
#                 nn.BatchNorm2d(out_channels),
#                 act_fn,
#                 nn.MaxPool2d(kernel_size=(2, 2))
#             ])
#             in_channels = out_channels
#         self.cnn2d = nn.Sequential(*conv2d_layers)

#         # Calculate flattened feature size after CNNs for FC input
#         with torch.no_grad():
#             dummy = torch.zeros(1, number_channels, input_time_length)
#             dummy = self.cnn1d(dummy)
#             dummy = dummy.unsqueeze(1)
#             dummy = self.cnn2d(dummy)
#             batch, c, ch, t = dummy.shape
#             flattened_size = c * ch * t

#         # Fully Connected Classification Layer
#         self.fc = nn.Sequential(
#             nn.Flatten(),
#             nn.Linear(flattened_size, fc_hidden),
#             act_fn,
#             nn.Dropout(fc_dropout),
#             nn.Linear(fc_hidden, number_classes)
#         )

#     def forward(self, x):
#         x = self.cnn1d(x)  # (batch, num_temporal_filters, time')
#         x = x.unsqueeze(1) # (batch, 1, channels, time')
#         x = self.cnn2d(x)  # (batch, last_conv2d_channel, ch', t')
#         x = self.fc(x)     # (batch, number_classes)
#         return x

#     def predict(self, loader, device=None):
#         preds = []
#         with torch.no_grad():
#             for batch in loader:
#                 if isinstance(batch, (list, tuple)):
#                     batch_x = batch[0]
#                 else:
#                     batch_x = batch
#                 if device is not None:
#                     batch_x = batch_x.to(device)
#                 outputs = self.forward(batch_x)
#                 batch_preds = torch.argmax(outputs, dim=1).cpu().numpy()
#                 preds.extend(batch_preds)
#         return np.array(preds)
        
#     def evaluate(self, y_pred, y_true):
#         if isinstance(y_true, torch.Tensor):
#             y_true = y_true.cpu().numpy()
#         if isinstance(y_pred, torch.Tensor):
#             y_pred = y_pred.cpu().numpy()
#         results = {}
#         results["overall_accuracy"] = accuracy_score(y_true, y_pred)
#         results["overall_F1"] = f1_score(y_true, y_pred, average='macro', zero_division=0)
#         results["overall_recall"] = recall_score(y_true, y_pred, average='macro', zero_division=0)
#         results["overall_precision"] = precision_score(y_true, y_pred, average='macro', zero_division=0)
#         if self.tasks is not None and len(self.tasks) > 1:
#             for idx, task in enumerate(self.tasks):
#                 mask = (y_true == idx)
#                 if np.sum(mask) == 0:
#                     results[f"{task}_accuracy"] = np.nan
#                     results[f"{task}_F1"] = np.nan
#                     results[f"{task}_recall"] = np.nan
#                     results[f"{task}_precision"] = np.nan
#                 else:
#                     results[f"{task}_accuracy"] = accuracy_score(y_true[mask], y_pred[mask])
#                     results[f"{task}_F1"] = f1_score(y_true[mask], y_pred[mask], average='macro', zero_division=0)
#                     results[f"{task}_recall"] = recall_score(y_true[mask], y_pred[mask], average='macro', zero_division=0)
#                     results[f"{task}_precision"] = precision_score(y_true[mask], y_pred[mask], average='macro', zero_division=0)
#         return results
    
# # Add this helper class at the top of your Model.py
# class Permute(nn.Module):
#     def __init__(self, *dims):
#         super().__init__()
#         self.dims = dims
#     def forward(self, x):
#         return x.permute(*self.dims)

# class MultiviewAttention(nn.Module):
#     def __init__(
#         self,
#         window_size,
#         conv1_filters,
#         conv2_filters,
#         conv3_filters,
#         conv1_kernel_shape,
#         conv2_kernel_shape,
#         conv3_kernel_shape,
#         padding1,
#         padding2,
#         padding3,
#         conv1_activation,
#         conv2_activation,
#         conv3_activation,
#         dense_nodes,
#         dense_activation,
#         depth,
#         lstm1_cells,
#         lstm2_cells,
#         dense3_nodes,
#         dense3_activation,
#         dropout_rate=0.3,  # Suggest lower default dropout
#         number_classes=4,
#         number_channels=248,
#         mesh_rows=20,
#         mesh_columns=21,
#         split_size=None,
#         transformer_nhead=4,
#         transformer_num_layers=1

#     ):
#         super().__init__()
#         self.window_size = window_size
#         self.number_classes = number_classes
#         self.number_channels = number_channels
#         self.mesh_rows = mesh_rows
#         self.mesh_columns = mesh_columns
#         self.depth = depth
#         self.tasks = ["resting", "math", "memory", "motor"]
#         self.split_size = split_size
#         self.dropout_rate = dropout_rate
#         self.dropout = nn.Dropout(self.dropout_rate)
#         self.transformer_nhead = transformer_nhead
#         self.transformer_num_layers = transformer_num_layers

#         # Activation functions
#         activations = {
#             'relu': nn.ReLU(),
#             'elu': nn.ELU(),
#             'gelu': nn.GELU(),
#             'leakyrelu': nn.LeakyReLU(),
#             'tanh': nn.Tanh()
#         }
        # act1 = activations.get(conv1_activation.lower(), nn.ReLU())
        # act2 = activations.get(conv2_activation.lower(), nn.ReLU())
        # act3 = activations.get(conv3_activation.lower(), nn.ReLU())
        # act_dense = activations.get(dense_activation.lower(), nn.ReLU())
        # act_dense3 = activations.get(dense3_activation.lower(), nn.ReLU())

#         # CNN branches for each window (up to flatten)
#         self.cnn_branches = nn.ModuleList([
#             nn.Sequential(
#                 nn.Conv2d(depth, conv1_filters, kernel_size=conv1_kernel_shape, padding=padding1),
#                 nn.BatchNorm2d(conv1_filters),
#                 act1,
#                 nn.Dropout(self.dropout_rate),
                
#                 nn.Conv2d(conv1_filters, conv2_filters, kernel_size=conv2_kernel_shape, padding=padding2),
#                 nn.BatchNorm2d(conv2_filters),
#                 act2,
#                 # Removed dropout here for less regularization
                
#                 nn.Conv2d(conv2_filters, conv3_filters, kernel_size=conv3_kernel_shape, padding=padding3),
#                 nn.BatchNorm2d(conv3_filters),
#                 act3,
#                 nn.AdaptiveAvgPool2d((1, 1)),
#                 nn.Flatten(start_dim=1)
#             ) for _ in range(window_size)
#         ])

#         # Dynamically compute flattened size for FC layer
#         with torch.no_grad():
#             dummy = torch.zeros(1, depth, mesh_rows, mesh_columns)
#             dummy_out = self.cnn_branches[0](dummy)
#             flattened_size = dummy_out.shape[1]

#         # Dense after CNN branch
#         self.cnn_dense = nn.ModuleList([
#             nn.Sequential(
#                 nn.Linear(flattened_size, dense_nodes),
#                 act_dense,
#                 nn.Dropout(self.dropout_rate) 
#             ) for _ in range(window_size)
#         ])

#         # Self-attention mechanism
#         self.attention = nn.MultiheadAttention(embed_dim=dense_nodes, num_heads=transformer_nhead, batch_first=True)

#         # LSTM branches for each window
#         self.lstm_dense = nn.ModuleList([
#             nn.Sequential(
#                 nn.Linear(depth, dense_nodes),
#                 act_dense
#             ) for _ in range(window_size)
#         ])

#         self.lstm_temporal_attention = nn.Sequential(
#             nn.Linear(dense_nodes, 64),
#             nn.Tanh(),
#             nn.Linear(64, 1)
#         )

#         # LSTM layers
#         self.lstm1 = nn.LSTM(dense_nodes, lstm1_cells, batch_first=True)
#         self.lstm2 = nn.LSTM(lstm1_cells, lstm2_cells, batch_first=True)

#         # Transformer encoder block (NEW)
#         self.transformer_encoder = nn.TransformerEncoder(
#             nn.TransformerEncoderLayer(d_model=dense_nodes, nhead=transformer_nhead, batch_first=True, dropout=self.dropout_rate),
#             num_layers=transformer_num_layers
#         )

#         # Dense after LSTM
#         self.dense3 = nn.Linear(lstm2_cells, dense3_nodes)
#         self.dense3_act = act_dense3

#         # Final layers
#         self.final_add = nn.Linear(dense_nodes, dense_nodes)  # for addition, if needed
#         # +dense_nodes for transformer output
#         self.final_out = nn.Linear(dense3_nodes + dense_nodes + dense_nodes + dense_nodes, number_classes)

#     def forward(self, x_cnn_list, x_lstm_list):
#         cnn_outs = []
#         for i, (branch, dense) in enumerate(zip(self.cnn_branches, self.cnn_dense)):
#             out = branch(x_cnn_list[i])
#             out = dense(out)
#             cnn_outs.append(out)
        
#         lstm_in = []
#         for branch, x_lstm in zip(self.lstm_dense, x_lstm_list):
#             out = branch(x_lstm)  # (batch, split_size, dense_nodes)
#             attn_weights = self.lstm_temporal_attention(out).squeeze(-1)  # (batch, split_size)
#             attn_weights = F.softmax(attn_weights, dim=1).unsqueeze(-1)   # (batch, split_size, 1)
#             out = (out * attn_weights).sum(dim=1)  # Weighted sum: (batch, dense_nodes)
#             lstm_in.append(out)
#         lstm_in = torch.stack(lstm_in, dim=1)  # (batch, window_size, dense_nodes)
        
#         # Self-attention across windows
#         attn_output, _ = self.attention(lstm_in, lstm_in, lstm_in)
#         attn_feat = self.dropout(attn_output.mean(dim=1))

#         # Transformer encoder (NEW)
#         transformer_out = self.transformer_encoder(lstm_in)
#         transformer_feat = self.dropout(transformer_out.mean(dim=1))

#         # LSTM
#         lstm_out, _ = self.lstm1(lstm_in)
#         lstm_out, _ = self.lstm2(lstm_out)
#         lstm_feat = self.dense3_act(self.dense3(lstm_out[:, -1, :]))
#         lstm_feat = self.dropout(lstm_feat)

#         added = self.dropout(torch.stack(cnn_outs, dim=0).sum(dim=0))
#         final = torch.cat([lstm_feat, added, attn_feat, transformer_feat], dim=-1)
#         final = self.dropout(final)

#         out = self.final_out(final)
#         return out

#     def predict(self, loader, device=None):
#         preds = []
#         self.eval()
#         with torch.no_grad():
#             for batch in loader:
#                 if isinstance(batch, (list, tuple)):
#                     batch_x_cnn, batch_x_lstm = batch[0], batch[1]
#                 else:
#                     batch_x_cnn, batch_x_lstm = batch, batch
#                 if device is not None:
#                     batch_x_cnn = [x.to(device) for x in batch_x_cnn]
#                     batch_x_lstm = [x.to(device) for x in batch_x_lstm]
#                 outputs = self.forward(batch_x_cnn, batch_x_lstm)
#                 batch_preds = torch.argmax(outputs, dim=1).cpu().numpy()
#                 preds.extend(batch_preds)
#         return np.array(preds)

#     def evaluate(self, y_pred, y_true):
#         if isinstance(y_true, torch.Tensor):
#             y_true = y_true.cpu().numpy()
#         if isinstance(y_pred, torch.Tensor):
#             y_pred = y_pred.cpu().numpy()
#         results = {}
#         results["overall_accuracy"] = accuracy_score(y_true, y_pred)
#         results["overall_F1"] = f1_score(y_true, y_pred, average='macro', zero_division=0)
#         results["overall_recall"] = recall_score(y_true, y_pred, average='macro', zero_division=0)
#         results["overall_precision"] = precision_score(y_true, y_pred, average='macro', zero_division=0)
#         if hasattr(self, "tasks") and self.tasks is not None and len(self.tasks) > 1:
#             for idx, task in enumerate(self.tasks):
#                 mask = (y_true == idx)
#                 if np.sum(mask) == 0:
#                     results[f"{task}_accuracy"] = np.nan
#                     results[f"{task}_F1"] = np.nan
#                     results[f"{task}_recall"] = np.nan
#                     results[f"{task}_precision"] = np.nan
#                 else:
#                     results[f"{task}_accuracy"] = accuracy_score(y_true[mask], y_pred[mask])
#                     results[f"{task}_F1"] = f1_score(y_true[mask], y_pred[mask], average='macro', zero_division=0)
#                     results[f"{task}_recall"] = recall_score(y_true[mask], y_pred[mask], average='macro', zero_division=0)
#                     results[f"{task}_precision"] = precision_score(y_true[mask], y_pred[mask], average='macro', zero_division=0)
#         return results

# class EEGNet(nn.Module):
#     def __init__(self, nb_classes, Channels, Samples, conv1_filters, conv2_filters,
#                  dropoutRate=0.5, kernLength=64,  
#                  depth=2, norm_rate=0.25, dropoutType='Dropout'):
#         super(EEGNet, self).__init__()

#         if dropoutType == 'SpatialDropout2D':
#             self.dropout = nn.Dropout2d
#         elif dropoutType == 'Dropout':
#             self.dropout = nn.Dropout
#         else:
#             raise ValueError("dropoutType must be one of 'SpatialDropout2D' or 'Dropout'.")

#         self.firstconv = nn.Sequential(
#             nn.Conv2d(1, conv1_filters, kernel_size=(1, kernLength), padding='same', bias=False),
#             nn.BatchNorm2d(conv1_filters)
#         )
        

#         self.depthwiseConv = nn.Sequential(
#             nn.Conv2d(conv1_filters, conv1_filters * depth, kernel_size=(Channels, 1), groups=conv1_filters, bias=False),
#             nn.BatchNorm2d(conv1_filters * depth),
#             nn.ELU(),
#             nn.AvgPool2d(kernel_size=(1, 4)),
#             self.dropout(dropoutRate)
#         )

#         self.separableConv = nn.Sequential(
#             nn.Conv2d(conv1_filters * depth, conv1_filters * depth, kernel_size=(1, 16), groups=conv1_filters * depth, padding='same', bias=False),
#             nn.Conv2d(conv1_filters * depth, conv2_filters, kernel_size=1, bias=False),
#             nn.BatchNorm2d(conv2_filters),
#             nn.ELU(),
#             nn.AvgPool2d(kernel_size=(1, 8)),
#             self.dropout(dropoutRate)
#         )

#         # Compute output size after conv/pool layers
#         self._output_size = self._get_flattened_size(Channels, Samples)

#         self.classifier = nn.Sequential(
#             nn.Linear(self._output_size, nb_classes),
#             nn.Softmax(dim=1)
#         )
      

#     def _get_flattened_size(self, Channels, Samples):
#         with torch.no_grad():
#             x = torch.zeros(1, 1, Channels, Samples)
#             x = self.firstconv(x)
#             x = self.depthwiseConv(x)
#             x = self.separableConv(x)
#             return x.view(1, -1).shape[1]

#     def forward(self, x):
#         x = self.firstconv(x)
#         x = self.depthwiseConv(x)
#         x = self.separableConv(x)
#         x = x.view(x.size(0), -1)
#         x = self.classifier(x)
#         return x


# class SimpleAttention(nn.Module):
#     def __init__(self, input_dim):
#         super().__init__()
#         self.attn = nn.Linear(input_dim, 1)

#     def forward(self, x):
#         # x: (batch, seq_len, input_dim)
#         weights = F.softmax(self.attn(x), dim=1)  # (batch, seq_len, 1)
#         return (x * weights).sum(dim=1)  # (batch, input_dim)


# class MultiviewAttentionOriginal(nn.Module):
#     def __init__(self, in_channels, mesh_rows, mesh_columns, window_size, cnn_filters, lstm_hidden, num_classes, dropout_rate=0.3):
#         super().__init__()
#         # 2D CNN for spatial feature extraction per window
#         self.cnn = nn.Sequential(
#             nn.Conv2d(in_channels, cnn_filters, kernel_size=3, padding=1),
#             nn.BatchNorm2d(cnn_filters),
#             nn.ReLU(),
#             nn.Dropout2d(dropout_rate),
#             nn.Conv2d(cnn_filters, cnn_filters, kernel_size=3, padding=1),
#             nn.BatchNorm2d(cnn_filters),
#             nn.ReLU(),
#             nn.Dropout2d(dropout_rate),
#             nn.AdaptiveAvgPool2d((1, 1))  # Pool to (1,1) per filter
#         )
#         # LSTM for temporal modeling across windows
#         self.lstm = nn.LSTM(cnn_filters, lstm_hidden, batch_first=True)
#         self.fc = nn.Linear(lstm_hidden, num_classes)

#     def forward(self, x):
#         # x: (batch, window_size, in_channels, mesh_rows, mesh_columns)
#         batch, windows, C, H, W = x.shape
#         x = x.view(batch * windows, C, H, W)
#         x = self.cnn(x)  # (batch*windows, cnn_filters, 1, 1)
#         x = x.view(batch, windows, -1)  # (batch, windows, cnn_filters)
#         x, _ = self.lstm(x)  # (batch, windows, lstm_hidden)
#         x = x[:, -1, :]  # Take last time step
#         x = self.fc(x)
#         return x
    
#     def predict(self, loader, device):
#         self.eval()
#         preds = []
#         with torch.no_grad():
#             for batch in loader:
#                 batch_x_cnn, batch_x_lstm, _ = batch  # unpack all, but only use batch_x_cnn
#                 batch_x_cnn = [x.to(device) for x in batch_x_cnn]
#                 outputs = self.forward(batch_x_cnn)  # Only pass batch_x_cnn!
#                 batch_preds = torch.argmax(outputs, dim=1).cpu().numpy()
#                 preds.extend(batch_preds)
#         return np.array(preds)

#     def evaluate(self, y_pred, y_true):
#         if isinstance(y_true, torch.Tensor):
#             y_true = y_true.cpu().numpy()
#         if isinstance(y_pred, torch.Tensor):
#             y_pred = y_pred.cpu().numpy()
#         results = {}
#         results["overall_accuracy"] = accuracy_score(y_true, y_pred)
#         results["overall_F1"] = f1_score(y_true, y_pred, average='macro', zero_division=0)
#         results["overall_recall"] = recall_score(y_true, y_pred, average='macro', zero_division=0)
#         results["overall_precision"] = precision_score(y_true, y_pred, average='macro', zero_division=0)
#         if hasattr(self, "tasks") and self.tasks is not None and len(self.tasks) > 1:
#             for idx, task in enumerate(self.tasks):
#                 mask = (y_true == idx)
#                 if np.sum(mask) == 0:
#                     results[f"{task}_accuracy"] = np.nan
#                     results[f"{task}_F1"] = np.nan
#                     results[f"{task}_recall"] = np.nan
#                     results[f"{task}_precision"] = np.nan
#                 else:
#                     results[f"{task}_accuracy"] = accuracy_score(y_true[mask], y_pred[mask])
#                     results[f"{task}_F1"] = f1_score(y_true[mask], y_pred[mask], average='macro', zero_division=0)
#                     results[f"{task}_recall"] = recall_score(y_true[mask], y_pred[mask], average='macro', zero_division=0)
#                     results[f"{task}_precision"] = precision_score(y_true[mask], y_pred[mask], average='macro', zero_division=0)
#         return results
