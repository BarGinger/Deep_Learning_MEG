"""
file: utils.py

Utility functions for loading and processing EEG data from h5py files.
Last updated: 19/05/2025
"""

# Import necessary libraries
import h5py
import glob
import os
import re
import pandas as pd
import numpy as np
from scipy.io import loadmat
from IPython.display import display
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.cuda.amp import GradScaler, autocast  # Add these imports for mixed precision training
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler, TensorDataset
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, roc_curve, auc, accuracy_score, f1_score, recall_score, precision_score
import sklearn.preprocessing
from tqdm.auto import tqdm
import seaborn
from enum import Enum
import shutil
import subprocess
import zipfile
import random
import optuna
import json
from Model import MultiviewDataset, MultiviewAttention
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.model_selection import StratifiedKFold
import matplotlib.pyplot as plt
from sklearn.model_selection import GroupKFold


# Gloval variables
debug_mode = False  # controls whether to print debug information
TASKS = ["resting", "math", "memory", "motor"]
METRICS = ["accuracy", "F1", "recall", "precision"]
# selced model to train
ModelClass = MultiviewAttention

# functions
def get_dataset_from_h5py(filename_path):
    """
        Get the dataset from the h5py file.
        Parameters:
        filename_path (str): The path to the h5py file.
        Returns:
        matrix (numpy.ndarray): The dataset as a numpy array.
    """

    matrix = None
    try:
        with h5py.File(filename_path, 'r') as f:
            # List all datasets or groups in the file
            avilable_datasets = list(f.keys())
            if len(avilable_datasets) == 0:
                print("No datasets or groups found in the file.")
            elif debug_mode:
                print(f"Available datasets file {filename_path}: {avilable_datasets}")

            dataset_name = avilable_datasets[0]
            # Check if the dataset exists in the file
            if dataset_name in f:
                matrix = f.get(dataset_name)[()]
            else:
                print(f"Dataset '{dataset_name}' not found in the file.")
    except OSError as e:
        print(f"Error opening file {filename_path}: {e}")
    except Exception as e:
        print(f"An error occurred: {e}")

    return matrix

def load_dataset(which_dataset="intra", data_dir="./Data", scaler_type="Z_SCORE_LOCAL", downsample_rate=0):
    ## load the chosen dataset/s
    train_data, train_df = None, None
    test_data, test_df = None, None

    print(f"Loading {which_dataset} dataset...")
    if which_dataset == "intra":
        # INTRA TRAIN
        intra_train_data_scaled, intra_train_df = try_load_scaled("intra_train", scaler_type)
        if intra_train_data_scaled is None:
            intra_train_folder = os.path.join(data_dir, "Intra", "train")
            intra_train_data, intra_train_df = load_eeg_data_and_metadata(intra_train_folder, split="train")
            intra_train_data_scaled, intra_train_df = scale_and_cache(
                intra_train_data, intra_train_df, "intra_train", scaler_type, downsample_factor=downsample_rate)

        # INTRA TEST
        intra_test_data_scaled, intra_test_df = try_load_scaled("intra_test", scaler_type)
        if intra_test_data_scaled is None:
            intra_test_folder = os.path.join(data_dir, "Intra", "test")
            intra_test_data, intra_test_df = load_eeg_data_and_metadata(intra_test_folder, split="test")
            intra_test_data_scaled, intra_test_df = scale_and_cache(
                intra_test_data, intra_test_df, "intra_test", scaler_type, downsample_factor=downsample_rate)
        
        train_data, train_df = intra_train_data_scaled, intra_train_df
        test_data, test_df = intra_test_data_scaled, intra_test_df
    if which_dataset == "cross":
        # CROSS TRAIN
        cross_train_data_scaled, cross_train_df = try_load_scaled("cross_train", scaler_type)
        if cross_train_data_scaled is None:
            cross_train_folder = os.path.join(data_dir, "Cross", "train")
            cross_train_data, cross_train_df = load_eeg_data_and_metadata(cross_train_folder, split="train")
            cross_train_data_scaled, cross_train_df = scale_and_cache(
                cross_train_data, cross_train_df, "cross_train", scaler_type, downsample_factor=downsample_rate)

        # CROSS TEST (list of datasets)
        cross_test_data_scaled = []
        cross_test_df_list = []
        for i in range(1, 4):
            prefix = f"cross_test{i}"
            scaled, meta = try_load_scaled(prefix, scaler_type)
            if scaled is None:
                folder = os.path.join(data_dir, "Cross", f"test{i}")
                data, df = load_eeg_data_and_metadata(folder, split=f"test{i}")
                scaled, meta = scale_and_cache(data, df, prefix, scaler_type, downsample_factor=downsample_rate)
            cross_test_data_scaled.append(scaled)
            cross_test_df_list.append(meta)
        
        cross_test_df_merged = pd.concat(cross_test_df_list, ignore_index=True)
        cross_test_data_merged = torch.cat(cross_test_data_scaled, dim=0)

        train_data, train_df = cross_train_data_scaled, cross_train_df
        test_data, test_df = cross_test_data_merged, cross_test_df_merged

    print(f"Data for dataset {which_dataset} was loaded successfully")
    return train_data, train_df, test_data, test_df

def load_eeg_data_and_metadata(folder_path, split):
    import glob
    data = []
    meta = []
    file_paths = sorted(glob.glob(os.path.join(folder_path, "*.h5")))
    for file_path in file_paths:
        # Allow underscores in task name
        m = re.match(r".*[/\\](?P<task>.+)_(?P<patient_id>\d+)_(?P<trial>\d+)\.h5", file_path)
        if not m:
            print(f"Filename does not match expected pattern: {file_path}")
            continue
        task = m.group("task")
        patient_id = m.group("patient_id")
        trial = m.group("trial")
        dataset = get_dataset_from_h5py(file_path)
        if dataset is not None:
            data.append(torch.tensor(dataset, dtype=torch.float32))
            meta.append({
                "filename": os.path.basename(file_path),
                "dataset_name": task,
                "patient_id": patient_id,
                "trial": trial,
                "split": split,
                "task_type": task
            })
        else:
            print(f"Warning: {file_path} could not be loaded.")
    if len(data) == 0:
        raise RuntimeError(f"No data loaded from folder {folder_path}")
    df = pd.DataFrame(meta)
    return torch.stack(data), df


### normalizion and scaling and downsampling
def downsample(data, factor):
    """
    Downsample the input data by a given factor using average pooling.
    Args:
        data (torch.Tensor): Input data of shape (N, C, T).
        factor (int): Downsampling factor.
    Returns:
        torch.Tensor: Downsampled data of shape (N, C, T/factor).
    """
    
    pool = nn.AvgPool1d(factor)

    return pool(data)

class NormalizationType(Enum):
    MINMAX_LOCAL = 1
    Z_SCORE_LOCAL = 2
    Z_SCORE_TIMEWISE = 3

def scale_data(data, scaler_type):
    # data: torch.Tensor of shape (N, C, T)
    data_np = data.numpy() if isinstance(data, torch.Tensor) else data
    if scaler_type == NormalizationType.MINMAX_LOCAL:
        # Scale each sample independently
        scaled = []
        for x in data_np:
            scaler = sklearn.preprocessing.MinMaxScaler()
            scaled.append(scaler.fit_transform(x))
        return torch.tensor(np.stack(scaled))
    elif scaler_type == NormalizationType.Z_SCORE_LOCAL:
        scaled = []
        for x in data_np:
            scaler = sklearn.preprocessing.StandardScaler()
            scaled.append(scaler.fit_transform(x))
        return torch.tensor(np.stack(scaled))
    elif scaler_type == NormalizationType.Z_SCORE_TIMEWISE:
        mean = data.mean(dim=0)
        sd = data.std(dim=0)
        return (data - mean) / (sd + 1e-8)
    else:
        raise ValueError("Unknown scaler_type")

def save_tensor_and_df(tensor, df, prefix, scaler_type):
    # Save to Data/Normalized/scaler_type/
    scaler_str = scaler_type.name.lower()
    out_folder = os.path.join("Data", "Normalized", scaler_str)
    os.makedirs(out_folder, exist_ok=True)
    torch.save(tensor, os.path.join(out_folder, f"{prefix}_scaled.pt"))
    df.to_csv(os.path.join(out_folder, f"{prefix}_meta.csv"), index=False)

def load_tensor_and_df(prefix, scaler_type):
    scaler_str = scaler_type.name.lower()
    out_folder = os.path.join("Data", "Normalized", scaler_str)
    tensor = torch.load(os.path.join(out_folder, f"{prefix}_scaled.pt"), weights_only=False)
    df = pd.read_csv(os.path.join(out_folder, f"{prefix}_meta.csv"))
    return tensor, df

def scale_and_cache(data, df, prefix, scaler_type, downsample_factor):
    scaler_str = scaler_type.name.lower()
    out_folder = os.path.join("Data", "Normalized", scaler_str)
    tensor_path = os.path.join(out_folder, f"{prefix}_scaled.pt")
    df_path = os.path.join(out_folder, f"{prefix}_meta.csv")
    if os.path.exists(tensor_path) and os.path.exists(df_path):
        print(f"Loading cached: {tensor_path}")
        return load_tensor_and_df(prefix, scaler_type)
    print(f"Scaling and saving: {tensor_path}")
    scaled = scale_data(data, scaler_type)

    df['prefix'] = prefix
    df['scaler_type'] = scaler_str
    df['downsampled'] = downsample_factor > 0
    df['downsample_factor'] = downsample_factor
    df['task_type_cat'] = df['task_type'].astype('category').cat.codes

    if downsample:
        scaled = downsample(scaled, downsample_factor)

    save_tensor_and_df(scaled, df, prefix, scaler_type)
    return scaled, df

def try_load_scaled(prefix, scaler_type):
    scaler_str = scaler_type.name.lower()
    out_folder = os.path.join("Data", "Normalized", scaler_str)
    tensor_path = os.path.join(out_folder, f"{prefix}_scaled.pt")
    df_path = os.path.join(out_folder, f"{prefix}_meta.csv")
    if os.path.exists(tensor_path) and os.path.exists(df_path):
        print(f"Loading cached: {tensor_path}")
        return torch.load(tensor_path, weights_only=False), pd.read_csv(df_path)
    return None, None

def extract_sliding_windows(data_tensor, window_size=256, stride=128):
    """
    Efficiently extract sliding windows from EEG data using PyTorch's unfold.
    Args:
        data_tensor: (N, C, T) torch.Tensor
        window_size: int, length of each window
        stride: int, stride between windows
    Returns:
        windows: (num_windows_total, C, window_size) torch.Tensor
        sample_indices: (num_windows_total,) torch.Tensor
    """
    N, C, T = data_tensor.shape
    # Use unfold to get windows for each sample
    # Result: (N, C, num_windows, window_size)
    windows = data_tensor.unfold(dimension=2, size=window_size, step=stride)
    num_windows = windows.shape[2]
    # Reshape to (N * num_windows, C, window_size)
    windows = windows.permute(0, 2, 1, 3).reshape(-1, C, window_size)
    # Track which sample each window came from
    sample_indices = torch.arange(N).repeat_interleave(num_windows)
    return windows, sample_indices


def get_dataloader(
    data_tensor, df, batch_size=64, input_time_length=256, stride=128, shuffle=False, use_windows=True
):
    """
    Create a DataLoader for EEG/MEG data, with optional sliding windows.
    """
    if use_windows:
        X, sample_indices = extract_sliding_windows(data_tensor, window_size=input_time_length, stride=stride)
        y_full = df['task_type_cat'].values
        y = torch.tensor(y_full[sample_indices], dtype=torch.long)
    else:
        X = data_tensor.float()
        y_full = df['task_type_cat'].values
        y = torch.tensor(y_full, dtype=torch.long)
    loader = DataLoader(TensorDataset(X, y), batch_size=batch_size, shuffle=shuffle)
    return loader, X, y



def get_dataloaders(data_tensor, df, fold_index=None, n_splits=4, batch_size=64):
    labels = df["task_type_cat"].values  # or whichever target you use

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    splits = list(skf.split(np.zeros(len(labels)), labels))
    train_idx, val_idx = splits[fold_index]
    
    train_df = df.iloc[train_idx]
    val_df = df.iloc[val_idx]
    

    train_dataset = TensorDataset(data_tensor[train_idx], torch.tensor(labels[train_idx]))
    val_dataset = TensorDataset(data_tensor[val_idx], torch.tensor(labels[val_idx]))

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    return train_loader, val_loader, train_df, val_df

## tainging and logging related functions
def create_metrics_df():
    """
    Create a DataFrame to log training metrics.
    """
    metrics = ["accuracy", "F1", "recall", "precision"]
    tasks = ["overall", "resting", "math", "memory", "motor"]

    # --- Dynamically get all possible hyperparameters from the search space ---
    # Use a dummy trial and dummy train_data to extract all keys
    import optuna
    class DummyTrial:
        def suggest_categorical(self, name, choices): return choices[0]
        def suggest_int(self, name, low, high, step=1): return low
        def suggest_float(self, name, low, high, step=None, log=False): return low
    dummy_trial = DummyTrial()
    # Use a dummy tensor with shape (N, C, T)
    dummy_train_data = torch.zeros(1, 248, 212)  # 248 channels, 212 time (arbitrary but valid)
    try:
        hyperparams = get_hyperparameter_search_space(dummy_trial, dummy_train_data)
        used_hyperparameters = list(hyperparams.keys())
    except Exception as e:
        # fallback: use static list if dummy fails
        if ModelClass in [MultiviewAttention]:
            used_hyperparameters = [
                "window_size", "conv1_filters", "conv2_filters", "conv3_filters",
                "conv1_kernel_shape", "conv2_kernel_shape", "conv3_kernel_shape",
                "padding1", "padding2", "padding3",
                "conv1_activation", "conv2_activation", "conv3_activation",
                "dense_nodes", "dense_activation", "depth",
                "lstm1_cells", "lstm2_cells", "dense3_nodes", "dense3_activation",
                "number_classes", "number_channels", "mesh_rows", "mesh_columns", "split_size",
                "learning_rate", "batch_size", "dropout_rate", "optimizer", "weight_decay",
                "label_smoothing", "transformer_nhead", "transformer_num_layers"
            ]
        else:
            used_hyperparameters = [
                "number_classes", "num_temporal_filters", "temporal_kernel_size", "dropout_rate",
                "activation", "pool_size", "conv2d_channels", "transformer_d_model",
                "transformer_nhead", "transformer_dim_feedforward", "transformer_layers",
                "fc_hidden", "fc_dropout", "learning_rate", "batch_size",
                "input_time_length", "stride"
            ]

    # Generate columns for train, val, and test metrics
    columns = ["trial_id", "epoch"]
    for split in ["train", "val", "test"]:
        for task in tasks:
            for metric in metrics:
                columns.append(f"{split}_{task}_{metric}")

    # Add hyperparameter columns
    columns.extend(used_hyperparameters)

    # Create the DataFrame
    all_metrics_df = pd.DataFrame(columns=columns)

    if debug_mode:
        print("Generated columns:")
        print(all_metrics_df.columns)

    return all_metrics_df


def get_mesh_candidates(split_size):
    mesh_row_candidates = [r for r in range(5, min(50, split_size + 1)) if split_size % r == 0]
    mesh_row_candidates.sort(key=lambda r: abs(r - (split_size / r)))

    return mesh_row_candidates

def get_mesh_candidate(split_size, mesh_rows_index=None):
    mesh_row_candidates = get_mesh_candidates(split_size)

    valid_index = mesh_rows_index is not None and mesh_rows_index > -1 and mesh_rows_index < len(mesh_row_candidates)
    return mesh_row_candidates[mesh_rows_index] if valid_index else mesh_row_candidates[0]


def train_full_model(
    study,
    train_data,
    train_df,
    device,
    epochs=100,
    verbose=True,
    path="full_training_curves.png",
    patience=10,
    min_delta=1e-4
):
    """
    Trains the best model found by Optuna on the full training set and plots training curves.
    Adds early stopping based on training loss.

    Args:
        study: Optuna study object after optimization.
        train_data: Training data (numpy array or tensor).
        train_df: Training DataFrame with labels.
        device: Torch device.
        epochs: Number of epochs to train.
        verbose: Print progress if True.
        path: Path to save the training curves plot.
        patience: Number of epochs with no improvement to wait before stopping.
        min_delta: Minimum change to qualify as an improvement.

    Returns:
        best_model: Trained model.
        best_params: Parameters used.
        (fig, (ax1, ax2)): Matplotlib figure and axes for the training curves.
        train_accs, train_f1s, train_losses: Lists of metrics per epoch.
    """

    best_trial = study.best_trial
    best_params = best_trial.params.copy()
    best_params["number_channels"] = train_data.shape[1]
    best_params["depth"] = 1  # <-- set to 1 for new model
    best_params["conv1_filters"] = 128  # <-- set to 1 for new model
    best_params["conv2_filters"] = 256  # <-- set to 1 for new model
    best_params["number_classes"] = len(np.unique(train_df['task_type']))
    best_params["num_classes"] = len(np.unique(train_df['task_type']))

    # Recompute split_size if missing
    if "split_size" not in best_params:
        T = train_data.shape[2]
        best_params["split_size"] = T // best_params["window_size"]

    # Always set mesh_rows and mesh_columns for new model
    if "mesh_rows" not in best_params:
        best_params["mesh_rows"] = 8
    if "mesh_columns" not in best_params:
        best_params["mesh_columns"] = 31

    # Instantiate model
    best_model = ModelClass(**{k: v for k, v in best_params.items() if k in ModelClass.__init__.__code__.co_varnames}).to(device)

    # Dataloader
    if ModelClass.__name__ in ["MultiviewAttention"]:
        train_loader = get_multiview_dataloader(
            train_data, train_df,
            batch_size=best_params["batch_size"],
            window_size=best_params["window_size"],
            mesh_rows=best_params["mesh_rows"],
            mesh_columns=best_params["mesh_columns"],
            depth=best_params["depth"],
            shuffle=True,
            augment=True
        )
    else:
        train_loader, train_X, train_y = get_dataloader(
            data_tensor=train_data, df=train_df,
            batch_size=best_params["batch_size"],
            input_time_length=best_params.get("input_time_length", 256),
            stride=best_params.get("stride", 256),
            shuffle=True
        )

    train_accs, train_f1s, train_losses = [], [], []

    optimizer_name = best_params.get("optimizer", "adam")
    weight_decay = best_params.get("weight_decay", 1e-4)
    learning_rate = best_params.get("learning_rate", 0.001)
    label_smoothing = best_params.get("label_smoothing", 0.1)

    optimizer, scheduler, criterion = get_training_parts(
        best_model, lr=learning_rate, max_epochs=epochs,
        optimizer_name=optimizer_name,
        weight_decay=weight_decay,
        label_smoothing=label_smoothing
    )

    best_train_loss = float('inf')
    epochs_without_improvement = 0

    epochs_pbar = tqdm(range(epochs), desc="Full Training", unit="epoch")
    for epoch in epochs_pbar:
        train_preds, train_targets, train_loss = single_train_epoch(
            model=best_model, 
            train_loader=train_loader,
            optimizer=optimizer,
            criterion=criterion,
            device=device
        )
        train_metrics = best_model.evaluate(y_true=train_targets, y_pred=train_preds)
        train_accs.append(train_metrics["overall_accuracy"])
        train_f1s.append(train_metrics["overall_F1"])
        train_losses.append(train_loss)
        scheduler.step()

        # Update progress bar
        epochs_pbar.set_postfix(
            accuracy=train_metrics["overall_accuracy"],
            F1=train_metrics["overall_F1"],
            loss=train_loss
        )
        

        # Early stopping logic (on training loss)
        if train_loss < best_train_loss - min_delta:
            best_train_loss = train_loss
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
        if epochs_without_improvement >= patience:
            print(f"Early stopping triggered after {epoch+1} epochs (no train_loss improvement).")
            break

    total_train_acc = np.mean(train_accs)
    print(f"Total Train Accuracy: {total_train_acc:.4f}")

    fig, ax1 = plt.subplots(figsize=(10, 5))
    ax1.plot(train_accs, label="Train Accuracy", color='tab:blue')
    ax1.plot(train_f1s, label="Train F1", linestyle="--", color='tab:green')
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Accuracy / F1")
    ax1.tick_params(axis='y')
    ax1.legend(loc='upper left')

    ax2 = ax1.twinx()
    ax2.plot(train_losses, label="Train Loss", linestyle=":", color='tab:purple')
    ax2.set_ylabel("Loss")
    ax2.tick_params(axis='y')
    ax2.legend(loc='upper right')

    plt.title(f"Full Training Accuracy, F1 & Loss ({ModelClass.__name__}) Total Train Acc: {total_train_acc:.4f}")
    plt.grid(True)
    plt.savefig(path, bbox_inches='tight')
    plt.show()

    return best_model, best_params, (fig, (ax1, ax2)), train_accs, train_f1s, train_losses

def single_train_epoch(model, train_loader, optimizer, criterion, device):
    model.train()
    train_preds, train_targets = [], []
    train_loss = 0.0
    for batch in train_loader:
        if isinstance(model, MultiviewAttention):
            # expects both x_cnn_list and x_lstm_list
            x_cnn_list, x_lstm_list, y_batch = batch
            x_cnn_list = [x.to(device) for x in x_cnn_list]
            x_lstm_list = [x.to(device) for x in x_lstm_list]
            y_batch = y_batch.to(device)
            optimizer.zero_grad()
            outputs = model(x_cnn_list, x_lstm_list)
        # elif isinstance(model, MultiviewAttentionOriginal):
        #     # expects only x_cnn_list
        #     x_cnn_list, x_lstm_list, y_batch = batch
        #     x_cnn_list = [x.to(device) for x in x_cnn_list]
        #     y_batch = y_batch.to(device)
        #     optimizer.zero_grad()
        #     outputs = model(x_cnn_list)
        else:
            x_batch, y_batch = batch
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)
            optimizer.zero_grad()
            outputs = model(x_batch)
        loss = criterion(outputs, y_batch)
        loss.backward()
        optimizer.step()
        train_loss += loss.item() * y_batch.size(0)
        train_preds.extend(outputs.argmax(dim=1).detach().cpu().numpy())
        train_targets.extend(y_batch.detach().cpu().numpy())
    train_loss /= len(train_loader.dataset)
    train_preds = np.array(train_preds)
    train_targets = np.array(train_targets)
    return train_preds, train_targets, train_loss


def single_eval_epoch(model, val_loader, criterion, device):
    """
    Evaluate the model for one epoch on the validation or test data.
    """
    model.eval()
    val_preds, val_targets = [], []
    val_loss = 0.0
    with torch.no_grad():
        for batch in val_loader:
            if isinstance(model, MultiviewAttention):
                # expects both x_cnn_list and x_lstm_list
                x_cnn_list, x_lstm_list, y_batch = batch
                x_cnn_list = [x.to(device) for x in x_cnn_list]
                x_lstm_list = [x.to(device) for x in x_lstm_list]
                y_batch = y_batch.to(device)
                outputs = model(x_cnn_list, x_lstm_list)
            # elif isinstance(model, MultiviewAttentionOriginal):
            #     # expects only x_cnn_list
            #     x_cnn_list, x_lstm_list, y_batch = batch
            #     x_cnn_list = [x.to(device) for x in x_cnn_list]
            #     y_batch = y_batch.to(device)
            #     outputs = model(x_cnn_list)
            else:
                x_batch, y_batch = batch
                x_batch = x_batch.to(device)
                y_batch = y_batch.to(device)
                outputs = model(x_batch)
            loss = criterion(outputs, y_batch)
            val_loss += loss.item() * y_batch.size(0)
            val_preds.extend(outputs.argmax(dim=1).detach().cpu().numpy())
            val_targets.extend(y_batch.detach().cpu().numpy())
    val_loss /= len(val_loader.dataset)
    val_preds = np.array(val_preds)
    val_targets = np.array(val_targets)
    return val_preds, val_targets, val_loss

def multiview_collate_fn(batch):
        # batch: list of (x_cnn_list, x_lstm_list, y)
        batch_size = len(batch)
        window_size = len(batch[0][0])
        # x_cnn: list of [batch, depth, mesh_rows, mesh_columns]
        x_cnn_list = [torch.stack([sample[0][i] for sample in batch], dim=0) for i in range(window_size)]
        # x_lstm: list of [batch, split_size, depth]
        x_lstm_list = [torch.stack([sample[1][i] for sample in batch], dim=0) for i in range(window_size)]
        y = torch.stack([sample[2] for sample in batch], dim=0)
        return x_cnn_list, x_lstm_list, y

def get_multiview_dataloader(data_tensor, df, batch_size, window_size, mesh_rows, mesh_columns, depth, shuffle=False, augment=False):
    y_full = df['task_type_cat'].values
    y = torch.tensor(y_full, dtype=torch.long)
    dataset = MultiviewDataset(data_tensor, y, window_size, mesh_rows, mesh_columns, depth, augment=augment)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, collate_fn=multiview_collate_fn)
    return loader



def get_multiview_dataloaders(
    data_tensor, df, fold_index=None, n_splits=4, batch_size=64,
    window_size=4, mesh_rows=20, mesh_columns=21, depth=248, property_to_split_on='patient_id'
):
    """
    Like get_dataloaders, but for MultiviewDataset.
    Ensures each fold contains only unseen patients in validation.
    Returns: train_loader, val_loader, train_df, val_df
    """

    if property_to_split_on == 'patient_id':
        groups = df["patient_id"].values
        gkf = GroupKFold(n_splits=n_splits)
        splits = list(gkf.split(np.zeros(len(groups)), groups=groups))
        train_idx, val_idx = splits[fold_index]
    else: # elif property_to_split_on == 'task_type_cat':
        # Use task_type_cat for stratified split if groups are not sufficient
        groups = df['task_type_cat'].values
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
        splits = list(skf.split(np.zeros(len(groups)), groups))
        train_idx, val_idx = splits[fold_index]

    train_df2 = df.iloc[train_idx]
    val_df2 = df.iloc[val_idx]

    y_full = df['task_type_cat'].values
    y_train = torch.tensor(y_full[train_idx], dtype=torch.long)
    y_val = torch.tensor(y_full[val_idx], dtype=torch.long)

    train_dataset = MultiviewDataset(data_tensor[train_idx], y_train, window_size, mesh_rows, mesh_columns, depth, augment=True)
    val_dataset = MultiviewDataset(data_tensor[val_idx], y_val, window_size, mesh_rows, mesh_columns, depth, augment=False)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=multiview_collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=multiview_collate_fn)
    
    return train_loader, val_loader, train_df2, val_df2


def plot_performance(mean_train_acc, mean_val_acc, mean_train_loss, mean_val_loss, trail_index, path=None):
    """
    Plot the mean training and validation accuracy and loss over epochs.

    Parameters:
        mean_train_acc (list): List of mean training accuracies per epoch.
        mean_val_acc (list): List of mean validation accuracies per epoch.
        mean_train_loss (list): List of mean training losses per epoch.
        mean_val_loss (list): List of mean validation losses per epoch.
        trail_index (int): Index of the training attempt for labeling the plot.
        path (str, optional): If provided, saves the plot to this path instead of displaying it.

    Returns:
        None: Displays the plot.
    """

    epochs = range(1, len(mean_train_acc) + 1)
    fig, ax1 = plt.subplots(figsize=(10, 5))

    # Accuracy on left y-axis
    ax1.plot(epochs, mean_train_acc, label="Mean Train Accuracy", color='tab:blue')
    ax1.plot(epochs, mean_val_acc, label="Mean Val Accuracy", color='tab:orange')
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Accuracy")
    ax1.legend(loc='upper left')

    # Loss on right y-axis
    ax2 = ax1.twinx()
    ax2.plot(epochs, mean_train_loss, label="Mean Train Loss", linestyle=":", color='tab:purple')
    ax2.plot(epochs, mean_val_loss, label="Mean Val Loss", linestyle=":", color='tab:brown')
    ax2.set_ylabel("Loss")
    ax2.legend(loc='upper right')

    plt.title(f"Mean Train/Val Accuracy & Loss (Trial {trail_index}) Overall Accuracy: {mean_val_acc[-1]:.4f})")
    plt.grid(True)
    if path:
        plt.savefig(path, bbox_inches='tight')
    plt.show()
    


def train_model(
    model, train_loader, val_loader, attempt_index, epochs, lr, patience, device,
    all_metrics_df, hyperparams, task_names, fold_idx, min_delta=1e-4, best_model_path=None
):
    model.to(device)

    optimizer_name = hyperparams.get("optimizer", "adam")
    weight_decay = hyperparams.get("weight_decay", 1e-4)
    label_smoothing = hyperparams.get("label_smoothing", 0.1)
    optimizer, scheduler, criterion = get_training_parts(model, lr=lr, max_epochs=epochs,
                                                         optimizer_name=optimizer_name,
                                                         weight_decay=weight_decay,
                                                         label_smoothing=label_smoothing)

    best_val_loss = float('inf')
    best_val_acc = 0.0
    best_model_state = None
    epochs_without_improvement = 0

    # For plotting
    train_accs, val_accs = [], []
    train_f1s, val_f1s = [], []
    train_losses, val_losses = [], []

    epochs_pbar = tqdm(range(epochs), desc=f"Trial: {attempt_index}, Fold index: {fold_idx+1}", unit="epoch")

    for epoch in epochs_pbar:
        # --- Training ---
        train_preds, train_targets, train_loss = single_train_epoch(
            model, train_loader, optimizer, criterion, device
        )
        # --- Validation ---
        val_preds, val_targets, val_loss = single_eval_epoch(
            model, val_loader, criterion, device
        )

        # --- Metrics ---
        row = {"trial_id": attempt_index, "epoch": epoch + 1, "fold_index": fold_idx+1}
        train_metrics = model.evaluate(y_true=train_targets, y_pred=train_preds)
        val_metrics = model.evaluate(y_true=val_targets, y_pred=val_preds)

        train_accs.append(train_metrics["overall_accuracy"])
        val_accs.append(val_metrics["overall_accuracy"])
        train_f1s.append(train_metrics["overall_F1"])
        val_f1s.append(val_metrics["overall_F1"])
        train_losses.append(train_loss)
        val_losses.append(val_loss)

        # Fill DataFrame row for all columns
        for split, metrics_dict in zip(["train", "val"], [train_metrics, val_metrics]):
            for key, value in metrics_dict.items():
                col = f"{split}_{key}"
                row[col] = value
        for task in task_names:
            for metric in METRICS:
                row[f"test_{task}_{metric}"] = np.nan
        for k, v in hyperparams.items():
            row[k] = v
        row["train_loss"] = train_loss
        row["val_loss"] = val_loss
        all_metrics_df = pd.concat([all_metrics_df, pd.DataFrame([row])], ignore_index=True)

        # --- Early stopping on val_loss ---
        improved = (best_val_loss - val_loss) > min_delta
        if improved:
            best_val_loss = val_loss
            best_val_acc = val_metrics["overall_accuracy"]
            best_model_state = model.state_dict()
            epochs_without_improvement = 0
            # Save best model to disk if path is given
            if best_model_path is not None:
                torch.save(best_model_state, best_model_path)
        else:
            epochs_without_improvement += 1
        if epochs_without_improvement >= patience:
            print(f"Early stopping triggered after {epoch + 1} epochs (no val_loss improvement).")
            break
        scheduler.step()

        epochs_pbar.set_postfix(
            train_accuracy=train_metrics["overall_accuracy"],
            val_accuracy=val_metrics["overall_accuracy"],
            train_F1=train_metrics["overall_F1"],
            val_F1=val_metrics["overall_F1"],
            train_loss=train_loss,
            val_loss=val_loss
        )

    # Restore best model weights
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
    # Optionally, save best model at the end if not already saved
    if best_model_path is not None and best_model_state is not None:
        torch.save(best_model_state, best_model_path)

    # --- Plotting ---
    # --- Plotting with dual y-axis ---
    if debug_mode:
        fig, ax1 = plt.subplots(figsize=(10, 5))

        # Accuracy/F1 on left y-axis
        ax1.plot(train_accs, label="Train Accuracy", color='tab:blue')
        ax1.plot(val_accs, label="Val Accuracy", color='tab:orange')
        ax1.plot(train_f1s, label="Train F1", linestyle="--", color='tab:green')
        ax1.plot(val_f1s, label="Val F1", linestyle="--", color='tab:red')
        ax1.set_xlabel("Epoch")
        ax1.set_ylabel("Accuracy / F1")
        ax1.tick_params(axis='y')
        ax1.legend(loc='upper left')

        # Loss on right y-axis
        ax2 = ax1.twinx()
        ax2.plot(train_losses, label="Train Loss", linestyle=":", color='tab:purple')
        ax2.plot(val_losses, label="Val Loss", linestyle=":", color='tab:brown')
        ax2.set_ylabel("Loss")
        ax2.tick_params(axis='y')
        ax2.legend(loc='upper right')

        plt.title(f"Training/Validation Accuracy, F1 & Loss (Trial {attempt_index}), Fold {fold_idx+1})")
        plt.grid(True)
        plt.show()
    return best_val_acc, all_metrics_df, train_accs, val_accs, train_losses, val_losses

# --- Optuna workflow ---
def compute_conv2d_output_size(H, W, kernel_size, padding, stride=1, dilation=1):
    # Standard PyTorch Conv2d output size formula
    out_h = (H + 2*padding - dilation*(kernel_size[0]-1) - 1)//stride + 1
    out_w = (W + 2*padding - dilation*(kernel_size[1]-1) - 1)//stride + 1
    return out_h, out_w

def get_hyperparameter_search_space(trial, train_data):
    if ModelClass in [MultiviewAttention]:
        window_size = trial.suggest_categorical('window_size', [8, 16, 24, 32])
        T = train_data.shape[2]
        split_size = T // window_size

        ALL_KERNEL_SHAPES = [(3, 3), (5, 5)]
        ALL_PADDINGS = [1, 2]

        depth = 1
        mesh_rows = 8
        mesh_columns = 31

        hyperparams = {
            "window_size": window_size,
            "conv1_filters": 128,  # fixed, always best
            "conv2_filters": 256,  # fixed, always best
            "conv3_filters": trial.suggest_categorical('conv3_filters', [128, 256]),
            "conv1_kernel_shape": trial.suggest_categorical('conv1_kernel_shape', ALL_KERNEL_SHAPES),
            "conv2_kernel_shape": trial.suggest_categorical('conv2_kernel_shape', ALL_KERNEL_SHAPES),
            "conv3_kernel_shape": trial.suggest_categorical('conv3_kernel_shape', ALL_KERNEL_SHAPES),
            "padding1": trial.suggest_categorical('padding1', ALL_PADDINGS),
            "padding2": trial.suggest_categorical('padding2', ALL_PADDINGS),
            "padding3": trial.suggest_categorical('padding3', ALL_PADDINGS),
            "conv1_activation": trial.suggest_categorical('conv1_activation', ['gelu', 'leaky_relu']),
            "conv2_activation": trial.suggest_categorical('conv2_activation', ['relu', 'leaky_relu']),
            "conv3_activation": trial.suggest_categorical('conv3_activation', ['relu', 'elu']),
            "dense_nodes": trial.suggest_categorical('dense_nodes', [256, 512]),
            "dense_activation": trial.suggest_categorical('dense_activation', ['relu', 'gelu']),
            "depth": depth,
            "lstm1_cells": trial.suggest_categorical('lstm1_cells', [32, 64, 128]),
            "lstm2_cells": trial.suggest_categorical('lstm2_cells', [32, 64, 128]),
            "dense3_nodes": trial.suggest_categorical('dense3_nodes', [32, 64, 128]),
            "dense3_activation": trial.suggest_categorical('dense3_activation', ['relu', 'gelu']),
            "number_classes": len(np.unique(TASKS)),
            "number_channels": train_data.shape[1],
            "mesh_rows": mesh_rows,
            "mesh_columns": mesh_columns,
            "split_size": split_size,
            "learning_rate": trial.suggest_float('learning_rate', 4e-4, 2e-3, log=True),
            "batch_size": trial.suggest_categorical('batch_size', [2, 4, 8, 16]),
            "dropout_rate": trial.suggest_float('dropout_rate', 0.1, 0.25, step=0.05),
            "optimizer": trial.suggest_categorical('optimizer', ['adam', 'adamw', 'rmsprop']),
            "weight_decay": trial.suggest_float('weight_decay', 1e-7, 1e-4, log=True),
            "label_smoothing": trial.suggest_float('label_smoothing', 0.05, 0.15, step=0.05),
            "number_of_heads": trial.suggest_categorical('number_of_heads', [2, 4]),
        }
        # Pruning checks as before...
        for kernel_shape, padding, name in [
            (hyperparams["conv1_kernel_shape"], hyperparams["padding1"], "conv1"),
            (hyperparams["conv2_kernel_shape"], hyperparams["padding2"], "conv2"),
            (hyperparams["conv3_kernel_shape"], hyperparams["padding3"], "conv3"),
        ]:
            if (mesh_rows + 2 * padding) < kernel_shape[0] or (mesh_columns + 2 * padding) < kernel_shape[1]:
                raise optuna.TrialPruned(
                    f"{name}: (mesh_rows + 2*padding)={mesh_rows + 2*padding}, (mesh_columns + 2*padding)={mesh_columns + 2*padding}, "
                    f"kernel_shape={kernel_shape} (padding={padding})"
                )
        H, W = mesh_rows, mesh_columns
        for k, p, name in [
            (hyperparams["conv1_kernel_shape"], hyperparams["padding1"], "conv1"),
            (hyperparams["conv2_kernel_shape"], hyperparams["padding2"], "conv2"),
            (hyperparams["conv3_kernel_shape"], hyperparams["padding3"], "conv3"),
        ]:
            H, W = compute_conv2d_output_size(H, W, k, p)
            if H < 1 or W < 1:
                raise optuna.TrialPruned(
                    f"{name}: Output size after conv would be ({H}, {W}) with kernel {k} and padding {p}."
                )
            if H < k[0] or W < k[1]:
                raise optuna.TrialPruned(
                    f"{name}: Output size ({H}, {W}) too small for next kernel {k}."
                )
        return hyperparams
        
def get_training_parts(model, lr=1e-3, weight_decay=1e-4, label_smoothing=0.1, max_epochs=100, optimizer_name="adam"):
    if optimizer_name == "adam":
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    elif optimizer_name == "adamw":
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    elif optimizer_name == "rmsprop":
        optimizer = torch.optim.RMSprop(model.parameters(), lr=lr, weight_decay=weight_decay)
    else:
        raise ValueError(f"Unknown optimizer: {optimizer_name}")
    
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max_epochs)
    
    # Define your task names and desired weights
    task_weights_dict = {
        "resting": 1,
        "math": 4,
        "memory": 4, 
        "motor": 4
    }
    # Make sure TASKS is in the same order as your class indices
    class_weights = torch.tensor([task_weights_dict[task] for task in TASKS], dtype=torch.float32).to(next(model.parameters()).device)
    
    criterion = torch.nn.CrossEntropyLoss(label_smoothing=label_smoothing, weight=class_weights)
    return optimizer, scheduler, criterion


def objective(trial, all_metrics_df, train_data, train_df, device, n_splits=4, best_model_path=None,
              plot_path=None, property_to_split_on='task_type_cat'):
    # Define hyperparameters using trial
    hyperparams = get_hyperparameter_search_space(trial, train_data)

    all_train_accs = []
    all_val_accs = []
    all_train_losses = []
    all_val_losses = []
    val_accs = []

    for fold_idx in range(n_splits):
        if ModelClass in [MultiviewAttention]:
            train_loader, val_loader, train_df2, val_df2 = get_multiview_dataloaders(
                train_data, train_df,
                fold_index=fold_idx,
                n_splits=n_splits,
                batch_size=hyperparams["batch_size"],
                window_size=hyperparams["window_size"],
                mesh_rows=hyperparams["mesh_rows"],
                mesh_columns=hyperparams["mesh_columns"],
                depth=hyperparams["depth"],
                property_to_split_on=property_to_split_on
            )
        else:
            train_loader, val_loader, train_df2, val_df2 = get_dataloaders(
                data_tensor=train_data,
                df=train_df,
                fold_index=fold_idx,
                n_splits=n_splits,
                batch_size=hyperparams["batch_size"]
            )

        if debug_mode:
            print(f"Fold {fold_idx + 1}/{n_splits} - Hyperparameters: {hyperparams}")
            print("train task types counts:")
            print(np.bincount(train_df2['task_type'].astype('category').cat.codes))
            print("validation task types counts:")
            print(np.bincount(val_df2['task_type'].astype('category').cat.codes))

            print("Train patient_id counts:\n", train_df2["patient_id"].value_counts())
            print("Val patient_id counts:\n", val_df2["patient_id"].value_counts())

        model = ModelClass(**{k: v for k, v in hyperparams.items() if k in ModelClass.__init__.__code__.co_varnames}).to(device)

        trial_index = trial.number + 1

        best_val_acc, updated_metrics_df, train_accs, val_accs_fold, train_losses, val_losses = train_model(
            model, train_loader, val_loader,
            attempt_index=trial_index,
            epochs=30,
            lr=hyperparams["learning_rate"],
            patience=10,
            device=device,
            all_metrics_df=all_metrics_df,
            hyperparams=hyperparams,
            task_names=TASKS,
            fold_idx=fold_idx,
            best_model_path=best_model_path,
            min_delta=1e-3
        )      

        val_accs.append(best_val_acc)
        all_metrics_df = updated_metrics_df  # accumulate across folds

        all_train_accs.append(train_accs)
        all_val_accs.append(val_accs_fold)
        all_train_losses.append(train_losses)
        all_val_losses.append(val_losses)

    # Compute means across folds
    # Find the maximum number of epochs across all folds
    max_len = max(len(x) for x in all_train_accs)

    def pad_to_max(arrays, fill_value=np.nan):
        return [np.pad(a, (0, max_len - len(a)), constant_values=fill_value) for a in arrays]

    all_train_accs = pad_to_max(all_train_accs)
    all_val_accs = pad_to_max(all_val_accs)
    all_train_losses = pad_to_max(all_train_losses)
    all_val_losses = pad_to_max(all_val_losses)

    mean_train_acc = np.nanmean(all_train_accs, axis=0)
    mean_val_acc = np.nanmean(all_val_accs, axis=0)
    mean_train_loss = np.nanmean(all_train_losses, axis=0)
    mean_val_loss = np.nanmean(all_val_losses, axis=0)

    avg_val_acc = sum(val_accs) / len(val_accs)
    avg_val_loss = sum(val_losses) / len(val_losses)

    plot_performance(
        mean_train_acc, mean_val_acc, mean_train_loss, mean_val_loss,
        trail_index=trial_index,
        path=plot_path
    )

    return avg_val_acc, avg_val_loss, all_metrics_df


def evaluate_model_on_testset(model, test_data, test_df, device, batch_size=64, input_time_length=256, stride=128, path=None):
    model.eval()
    if isinstance(model, (MultiviewAttention)):
        test_loader = get_multiview_dataloader(
            test_data, test_df, batch_size=batch_size,
            window_size=model.window_size,
            mesh_rows=model.mesh_rows,
            mesh_columns=model.mesh_columns,
            depth=model.depth,
            shuffle=False
        )
        test_y = torch.tensor(test_df['task_type_cat'].values, dtype=torch.long)
    else:
        test_loader, _, test_y = get_dataloader(
            data_tensor=test_data, df=test_df,
            batch_size=batch_size, input_time_length=input_time_length,
            stride=stride, shuffle=False
        )
    y_pred_test = model.predict(test_loader, device=device)
    test_metrics = model.evaluate(y_pred=y_pred_test, y_true=test_y)

    df_test_result = pd.DataFrame({
        "y_true": np.array(test_y),
        "y_pred": np.array(y_pred_test),
        "patient_id": test_df['patient_id']

    })
    if path:
        df_test_result.to_csv(path, index=False)
    return test_metrics, df_test_result


def plot_per_patient_accuracy_bars(
    df_test_result,
    output_dir,
    which_dataset,
    model_name,
    timestamp,
    task_names=TASKS,
    style="dark_background"
):
    """
    Plots and saves per-patient and overall accuracy per task as grouped bar plots.

    Args:
        df_test_result (pd.DataFrame): DataFrame with columns 'y_true', 'y_pred', 'patient_id'.
        output_dir (str): Output directory for saving the plot.
        which_dataset (str): Dataset name for path.
        model_name (str): Model name for path.
        timestamp (str): Timestamp for path.
        task_names (list, optional): List of task names. If None, will use sorted unique values from df_test_result.
        style (str): Matplotlib style to use.
    """
    plt.style.use(style)

    if task_names is None:
        task_names = [str(t) for t in sorted(df_test_result['y_true'].unique())]

    # Compute overall accuracy per task
    overall_acc_per_task = df_test_result.groupby('y_true').apply(lambda x: (x['y_true'] == x['y_pred']).mean())
    per_patient_acc = df_test_result.groupby(['patient_id', 'y_true']).apply(lambda x: (x['y_true'] == x['y_pred']).mean()).unstack(0)
    overall_acc = (df_test_result['y_true'] == df_test_result['y_pred']).mean()

    tasks = sorted(df_test_result['y_true'].unique())
    patients = per_patient_acc.columns.tolist()
    n_tasks = len(tasks)
    n_bars = len(patients) + 1  # patients + overall

    fig, axs = plt.subplots(1, n_tasks, figsize=(5*n_tasks, 7), sharey=True)

    bar_colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f']
    overall_color = '#ffffff'
    task_title_colors = ['#00bfff', '#ff69b4', '#ffd700', '#00fa9a']  # blue, pink, gold, green

    for t_idx, t in enumerate(tasks):
        ax = axs[t_idx]
        # Per-patient bars
        for i, patient in enumerate(patients):
            acc = per_patient_acc[patient].reindex(tasks, fill_value=0)[t]
            bar = ax.bar(i, acc, color=bar_colors[i % len(bar_colors)], width=0.7, label=f'Patient {patient}' if t_idx == 0 else "", zorder=3)
            label = str(int(acc)) if np.isclose(acc, 1.0) else f"{acc:.2f}"
            ax.text(i, acc + 0.025, label, ha='center', va='bottom', fontsize=16, color='white', fontweight='bold')
        # Overall bar
        overall = overall_acc_per_task.reindex(tasks, fill_value=0)[t]
        bar = ax.bar(len(patients), overall, color=overall_color, edgecolor='black', width=0.7, label='Overall' if t_idx == 0 else "", zorder=3)
        label = str(int(overall)) if np.isclose(overall, 1.0) else f"{overall:.2f}"
        ax.text(len(patients), overall + 0.025, label, ha='center', va='bottom', fontsize=16, color='white', fontweight='bold')
        # Formatting
        ax.set_title(task_names[t], fontsize=28, fontweight='bold', color=task_title_colors[t_idx % len(task_title_colors)])
        ax.set_xticks(list(range(n_bars)))
        ax.set_xticklabels([f"P{p}" for p in patients] + ["Overall"], fontsize=18, fontweight='bold', rotation=30)
        ax.set_ylim(0, 1.18)
        ax.grid(axis='y', linestyle=':', alpha=0.5, zorder=0)
        ax.tick_params(axis='y', labelsize=18)
        if t_idx == 0:
            ax.set_ylabel('Accuracy', fontsize=28, fontweight='bold', color='white')

        if t_idx == n_tasks-1:
            ax.legend(fontsize=16, loc='upper left', bbox_to_anchor=(1.05, 1.0))

    fig.supxlabel('Patients', fontsize=28, fontweight='bold', color='white')
    fig.suptitle(f'Per-patient and Overall Accuracy per Task - Overall Accuracy: {overall_acc:.3f}', fontsize=30, fontweight='bold', color='skyblue')
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    path = f"{output_dir}/{which_dataset}/per_patient_accuracy_{model_name}_{timestamp}.png"
    plt.savefig(path, bbox_inches='tight')
    plt.show()
    return fig, axs, path, per_patient_acc, overall_acc_per_task


def plot_per_patient_accuracy_heatmap(
    per_patient_acc,
    overall_acc_per_task,
    df_test_result,
    output_dir,
    which_dataset,
    model_name,
    timestamp,
    task_names=TASKS,
    title="Per-patient Accuracy per Task Heatmap"
):
    """
    Plots and saves a heatmap of per-patient accuracy per task, including an overall row.

    Parameters:
        per_patient_acc (pd.DataFrame): DataFrame indexed by patient and task with accuracy values.
        overall_acc_per_task (pd.Series): Series indexed by task with overall accuracy per task.
        df_test_result (pd.DataFrame): DataFrame with 'y_true' column for task indices.
        output_dir (str): Output directory for saving the plot.
        which_dataset (str): Dataset name for path.
        model_name (str): Model name for path.
        timestamp (str): Timestamp for path.
        task_names (list, optional): List of task names. If None, will use sorted unique values from df_test_result.
        title (str): Plot title.
    """
    plt.style.use('default')

    if task_names is None:
        task_names = [str(t) for t in sorted(df_test_result['y_true'].unique())]
    tasks = sorted(df_test_result['y_true'].unique())
    patients = per_patient_acc.columns.tolist()
    heatmap_data = per_patient_acc.reindex(index=tasks, columns=patients, fill_value=np.nan).T.values  # (n_patients, n_tasks)
    overall_per_task = overall_acc_per_task.reindex(tasks, fill_value=np.nan).values  # (n_tasks,)

    # Add overall_per_task as a new row at the end
    heatmap_data_with_overall = np.vstack([heatmap_data, overall_per_task[None, :]])
    yticklabels = [str(p) for p in patients] + [r"$\bf{Overall}$"]  # bold and pretty overall label

    fig, ax = plt.subplots(figsize=(10, max(6, len(yticklabels)*0.5)))

    im = ax.imshow(heatmap_data_with_overall, cmap='Blues', vmin=0, vmax=1)

    # Show all ticks and label them
    ax.set_xticks(np.arange(len(tasks)))
    ax.set_yticks(np.arange(len(yticklabels)))
    ax.set_xticklabels([task_names[t] for t in tasks], fontsize=11, fontweight='bold')
    ax.set_yticklabels(yticklabels, fontsize=11, fontweight='bold')

    plt.setp(ax.get_xticklabels(), rotation=20, ha="right", rotation_mode="anchor")

    # Color only the last yticklabel ("Overall") in deepskyblue
    for tick in ax.get_yticklabels():
        tick.set_color("black")
    ax.get_yticklabels()[-1].set_color("deepskyblue")

    # Loop over data dimensions and create text annotations.
    for i in range(len(yticklabels)):
        for j in range(len(tasks)):
            val = heatmap_data_with_overall[i, j]
            if not np.isnan(val):
                label = str(int(val)) if np.isclose(val, 1.0) else f"{val:.2f}"
                # Make "Overall" row text bold and deepskyblue, others normal black
                if i == len(yticklabels)-1:
                    color = "deepskyblue"
                    weight = 'bold'
                else:
                    color = "black"
                    weight = 'normal'
                ax.text(j, i, label, ha="center", va="center", color=color, fontsize=9, fontweight=weight)

    ax.set_xlabel("Task", fontsize=14, fontweight='bold', color="darkblue")
    ax.set_ylabel("Patient", fontsize=14, fontweight='bold', color="darkblue")
    ax.set_title(title, color="darkblue", fontsize=15, fontweight='bold')
    # Make colorbar shorter (smaller height)
    cbar = fig.colorbar(im, ax=ax, fraction=0.025, pad=0.04)
    cbar.set_label('Accuracy', fontsize=12, fontweight="bold", color="darkred")
    plt.tight_layout()
    path = f"{output_dir}/{which_dataset}/per_patient_accuracy_heatmap_{model_name}_{timestamp}.png"
    plt.savefig(path, bbox_inches='tight')
    plt.show()
    return fig, ax, path


def evaluate_and_log_testset(
    study,
    best_model,
    best_params,
    test_data,
    test_df,
    device,
    all_metrics_df,
    model_csv_path,
    output_dir,
    which_dataset,
    model_name,
    timestamp,
    path=None
):
    """
    Evaluates the best model on the test set, updates metrics DataFrame, and saves results.

    Returns:
        test_metrics: dict of test metrics
        df_test_result: DataFrame with test results
        all_metrics_df: updated metrics DataFrame
    """
    if path is None:
        test_csv_path = f"{output_dir}/{which_dataset}/test_metrics_{model_name}_{timestamp}.csv"
    else:
        test_csv_path = path

    best_trial = study.best_trial

    test_metrics, df_test_result = evaluate_model_on_testset(
        best_model, test_data, test_df, device,
        batch_size=best_params.get("batch_size", 64),
        input_time_length=best_params.get("input_time_length", 256),
        stride=best_params.get("stride", 256),
        path=test_csv_path
    )

    print("Test metrics:", test_metrics, "\nBest trial ID:", best_trial.number + 1, '\n')

    # Add "test_" prefix to all columns in test_metrics
    test_metrics_prefixed = {f"test_{k}": v for k, v in test_metrics.items()}

    # Find the row for the best trial (trial_id matches best_trial.number + 1)
    row_idx = all_metrics_df[all_metrics_df["trial_id"] == best_trial.number + 1].index
    if len(row_idx) == 0:
        # fallback: use last row
        row_idx = [all_metrics_df.index[-1]]
    row_idx = row_idx[0]

    # Copy all columns in test_metrics_prefixed to the DataFrame row
    for col, value in test_metrics_prefixed.items():
        if col in all_metrics_df.columns:
            all_metrics_df.at[row_idx, col] = value

    # Save metrics
    all_metrics_df.to_csv(model_csv_path, index=False)
    print(f"All metrics saved to {model_csv_path}")
    print("Best hyperparameters:", best_params)
    print("Best validation accuracy:", study.best_value)

    return test_metrics, df_test_result, all_metrics_df