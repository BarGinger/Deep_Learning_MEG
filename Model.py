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


def print_model_summary(model, input_shapes, device='cpu'):
    """
    Print a summary of the model using torchsummary.
    Args:
        model: The PyTorch model instance.
        input_shapes: List of tuples, one for each input. E.g. [(depth, mesh_rows, mesh_columns), ...]
        device: 'cpu' or 'cuda'
    """
    model = model.to(device)
    if isinstance(input_shapes, tuple):
        input_shapes = [input_shapes]
    try:
        summary(model, input_shapes)
    except Exception as e:
        print("torchsummary failed:", e)
        print("Model architecture:\n", model)


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
        # Always re-create if shape mismatch
        if (self.rel_height is None) or (self.rel_height.shape[0] != 2*H-1):
            self.rel_height = nn.Parameter(
                torch.randn(2*H-1, self.dk // self.n_heads, device=device) * (self.dk // self.n_heads) ** -0.5,
                requires_grad=True
            )
        if (self.rel_width is None) or (self.rel_width.shape[0] != 2*W-1):
            self.rel_width = nn.Parameter(
                torch.randn(2*W-1, self.dk // self.n_heads, device=device) * (self.dk // self.n_heads) ** -0.5,
                requires_grad=True
            )

    def _relative_logits_1d(self, q, rel_k, axis_size, other_axis_size, transpose_mask=None):
        # q: (B, n_heads, H, W, d_head) or (B, n_heads, W, H, d_head)
        # rel_k: (2*axis_size-1, d_head)
        B, n_heads, A1, A2, d_head = q.shape  # A1 = axis_size, A2 = other_axis_size
        # print("q.shape:", q.shape)
        # print("rel_k.shape:", rel_k.shape)
        rel_logits = torch.einsum('bhlad,md->bhlam', q, rel_k)  # (B, n_heads, A1, A2, 2A1-1)
        rel_logits = rel_logits.permute(0, 3, 1, 2, 4)  # (B, A2, n_heads, A1, 2A1-1)
        rel_logits = rel_logits.contiguous().view(B * A2 * n_heads, A1, 2*A1-1)
        rel_logits = self._rel_to_abs(rel_logits)  # (B*A2*n_heads, A1, A1)
        rel_logits = rel_logits.view(B, A2, n_heads, A1, A1).permute(0, 2, 1, 3, 4)  # (B, n_heads, A2, A1, A1)
        rel_logits = rel_logits.reshape(B, n_heads, A2*A1, A1)
        # Instead of expanding and reshaping, repeat along the correct axis to get (B, n_heads, HW, HW)
        rel_logits = rel_logits.repeat(1, 1, 1, A2)  # (B, n_heads, A2*A1, A1*A2)
        rel_logits = rel_logits[:, :, :, :A2*A1]     # Crop if needed
        return rel_logits

    def _rel_to_abs(self, x):
        # print(f"x.shape before _rel_to_abs: {x.shape}")
        # x: (batch, L, 2L-1)
        batch, L, _ = x.shape
        # Pad to shift from relative to absolute indexing
        col_pad = torch.zeros((batch, L, 1), device=x.device, dtype=x.dtype)
        x = torch.cat([x, col_pad], dim=2)  # (batch, L, 2L)        
        flat_x = x.reshape(batch, -1)       # (batch, L*2L)
        flat_pad = torch.zeros((batch, L-1), device=x.device, dtype=x.dtype)
        flat_x_padded = torch.cat([flat_x, flat_pad], dim=1)  # (batch, L*2L + L-1)
        # print(f"flat_x_padded.shape: {flat_x_padded.shape}, target: ({batch}, {L+1}, {2*L-1})")
        final_x = flat_x_padded.reshape(batch, L+1, 2*L-1)
        final_x = final_x[:, :L, L-1:]
        return final_x  # (batch, L, L)
    
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
            rel_logits_h = self._relative_logits_1d(q, self.rel_height, H, W)
            # Width
            rel_logits_w = self._relative_logits_1d(q.permute(0, 1, 3, 2, 4), self.rel_width, W, H)
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
        lstm1_layers=1,
        lstm2_layers=1,
        dropout_rate=0.1,
    ):
        super().__init__()
        self.window_size = window_size
        self.depth = depth
        self.channels = number_channels
        self.mesh_rows = mesh_rows
        self.mesh_columns = mesh_columns
        self.number_classes = number_classes
        self.lstm1_layers = lstm1_layers
        self.lstm2_layers = lstm2_layers

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
        self.dropout_rate = dropout_rate

        dk = max(4, number_of_heads * 2)  # or another multiple, but must be divisible by number_of_heads
        dv = max(4, number_of_heads * 2)  # same logic

        # CNN branches
        self.cnn_branches = nn.ModuleList()
        self.cnn_dense = nn.ModuleList()
        for _ in range(window_size):
            branch = nn.Sequential(
                AugmentedConv2D(depth, conv1_filters, conv1_kernel_shape, dk=dk, dv=dv, n_heads=number_of_heads, relative=True),
                nn.BatchNorm2d(conv1_filters),
                self.act1,
                nn.Dropout(dropout_rate),  # Add dropout here
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
                    self.act_dense,
                    nn.Dropout(dropout_rate)  # Add dropout here
                )
            )

        # LSTM branches
        self.lstm_dense = nn.ModuleList([
            nn.Sequential(
                nn.Linear(number_channels, dense_nodes),
                self.act_dense
            ) for _ in range(window_size)
        ])
        self.lstm1 = nn.LSTM(dense_nodes, lstm1_cells, batch_first=True, bidirectional=True, num_layers=lstm1_layers, dropout=dropout_rate)
        self.lstm2 = nn.LSTM(2 * lstm1_cells, lstm2_cells, batch_first=True, bidirectional=True,  num_layers=lstm2_layers,  dropout=dropout_rate)
        self.final_dense = nn.Sequential(
            nn.Linear(2 * lstm2_cells, dense3_nodes),
            self.act_dense3,
            nn.Dropout(dropout_rate)  # Add dropout here
        )

        # Final classifier
        self.classifier = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(dense3_nodes + dense_nodes, number_classes)
        )
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