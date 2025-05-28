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
from Model import MEGClassifier, MultiviewAttention, MultiviewDataset
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.model_selection import StratifiedKFold
import matplotlib.pyplot as plt


# Gloval variables
debug_mode = False  # controls whether to print debug information
TASKS = ["resting", "math", "memory", "motor"]
METRICS = ["accuracy", "F1", "recall", "precision"]
# selced model to train
ModelClass = MultiviewAttention  # or MEGClassifier, etc.

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

    # Define the metrics and tasks
    metrics = ["accuracy", "F1", "recall", "precision"]
    tasks = ["overall", "resting", "math", "memory", "motor"]
    # For MultiviewAttention
    hyperparameters = [
        "window_size",
        "conv1_filters",
        "conv2_filters",
        "conv3_filters",
        "conv1_kernel_shape",
        "conv2_kernel_shape",
        "conv3_kernel_shape",
        "padding1",
        "padding2",
        "padding3",
        "conv1_activation",
        "conv2_activation",
        "conv3_activation",
        "dense_nodes",
        "dense_activation",
        "depth",
        "lstm1_cells",
        "lstm2_cells",
        "dense3_nodes",
        "dense3_activation",
        "number_classes",
        "number_channels",
        "mesh_rows",
        "mesh_columns",
        "split_size",
        "learning_rate",
        "batch_size",
        "dropout_rate"
    ]

    # For MEGClassifier (or similar, if used)
    hyperparameters_classifier = [
        "number_classes",
        "num_temporal_filters",
        "temporal_kernel_size",
        "dropout_rate",
        "activation",
        "pool_size",
        "number_classes",
        "conv2d_channels",
        "transformer_d_model",
        "transformer_nhead",
        "transformer_dim_feedforward",
        "transformer_layers",
        "fc_hidden",
        "fc_dropout",
        "learning_rate",
        "batch_size",
        "input_time_length",  # for windowing
        "stride"              # for windowing
    ]

    if ModelClass == MultiviewAttention:
        used_hyperparameters = hyperparameters
    else:
        used_hyperparameters = hyperparameters_classifier

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

    # Example: Print the generated columns
    if debug_mode:
        print("Generated columns:")
        print(all_metrics_df.columns)

    return all_metrics_df


# def single_train_epoch(model, train_loader, optimizer, criterion, device):
#     model.train()
#     train_preds, train_targets = [], []
#     for x_batch, y_batch in train_loader:
#         x_batch, y_batch = x_batch.to(device), y_batch.to(device)
#         optimizer.zero_grad()
#         outputs = model(x_batch)
#         loss = criterion(outputs, y_batch)
#         loss.backward()
#         optimizer.step()
#         train_preds.extend(outputs.argmax(dim=1).detach().cpu().numpy())
#         train_targets.extend(y_batch.detach().cpu().numpy())
#     train_preds = np.array(train_preds)
#     train_targets = np.array(train_targets)

#     return train_preds, train_targets


def train_full_model(
    model, train_loader, epochs, lr, patience, device, hyperparams, save_path=None
):
    """
    Train the model on the full training dataset for a specified number of epochs.  
    
    Parameters:
        model (torch.nn.Module): The model to train.
        train_loader (DataLoader): DataLoader for the training data.
        epochs (int): Number of epochs to train the model.
        lr (float): Learning rate for the optimizer.
        patience (int): Number of epochs with no improvement after which training will be stopped.
        device (torch.device): Device to run the training on (CPU or GPU).
        hyperparams (dict): Hyperparameters for the model.
        save_path (str, optional): Path to save the trained model. If None, model won't be saved.
    Returns:
        model (torch.nn.Module): The trained model.
    """
    optimizer, scheduler, criterion = get_training_parts(model, lr=lr)

    for epoch in range(epochs):
        single_train_epoch(model, train_loader, optimizer, criterion, device)

    # Save final model
    if save_path:
        torch.save(model.state_dict(), save_path)
    return model

def single_train_epoch(model, train_loader, optimizer, criterion, device):
    """
    Train the model for one epoch on the training data.
    
    Parameters:
        model (torch.nn.Module): The model to train.
        train_loader (DataLoader): DataLoader for the training data.
        optimizer (torch.optim.Optimizer): Optimizer for updating model parameters.
        criterion (torch.nn.Module): Loss function to compute the loss.
        device (torch.device): Device to run the training on (CPU or GPU).
    
    Returns:
        train_preds (np.ndarray): Predictions on the training data.
        train_targets (np.ndarray): True labels for the training data.
        train_loss (float): Average loss over the training epoch.
    """ 
    model.train()
    train_preds, train_targets = [], []
    train_loss = 0.0
    for batch in train_loader:
        if isinstance(model, MultiviewAttention):
            if isinstance(batch, (list, tuple)) and len(batch) == 3:
                x_cnn_list, x_lstm_list, y_batch = batch
                x_cnn_list = [x.to(device) for x in x_cnn_list]
                x_lstm_list = [x.to(device) for x in x_lstm_list]
            else:
                raise ValueError("Expected batch to be (x_cnn_list, x_lstm_list, y_batch) for MultiviewAttention.")
            y_batch = y_batch.to(device)
            optimizer.zero_grad()
            outputs = model(x_cnn_list, x_lstm_list)
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
    
    Parameters:
        model (torch.nn.Module): The model to evaluate.
        val_loader (DataLoader): DataLoader for the evaluation data.
        criterion (torch.nn.Module): Loss function to compute the loss.
        device (torch.device): Device to run the evaluation on (CPU or GPU).
    
    Returns:
        eval_preds (np.ndarray): Predictions on the evaluation data.
        eval_targets (np.ndarray): True labels for the evaluation data.
    """
    model.eval()
    val_preds, val_targets = [], []
    val_loss = 0.0
    with torch.no_grad():
        for batch in val_loader:
            if isinstance(model, MultiviewAttention):
                if isinstance(batch, (list, tuple)) and len(batch) == 3:
                    x_cnn_list, x_lstm_list, y_batch = batch
                    x_cnn_list = [x.to(device) for x in x_cnn_list]
                    x_lstm_list = [x.to(device) for x in x_lstm_list]
                else:
                    raise ValueError("Expected batch to be (x_cnn_list, x_lstm_list, y_batch) for MultiviewAttention.")
                y_batch = y_batch.to(device)
                outputs = model(x_cnn_list, x_lstm_list)
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

def get_multiview_dataloader(data_tensor, df, batch_size, window_size, mesh_rows, mesh_columns, depth, shuffle=False):
    y_full = df['task_type_cat'].values
    y = torch.tensor(y_full, dtype=torch.long)
    dataset = MultiviewDataset(data_tensor, y, window_size, mesh_rows, mesh_columns, depth)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, collate_fn=multiview_collate_fn)
    return loader


def get_multiview_dataloaders(
    data_tensor, df, fold_index=None, n_splits=4, batch_size=64,
    window_size=4, mesh_rows=20, mesh_columns=21, depth=248
):
    """
    Like get_dataloaders, but for MultiviewDataset.
    Returns: train_loader, val_loader, train_df, val_df
    """
    labels = df["task_type_cat"].values
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    splits = list(skf.split(np.zeros(len(labels)), labels))
    train_idx, val_idx = splits[fold_index]

    train_df2 = df.iloc[train_idx]
    val_df2 = df.iloc[val_idx]

    y_full = df['task_type_cat'].values
    y_train = torch.tensor(y_full[train_idx], dtype=torch.long)
    y_val = torch.tensor(y_full[val_idx], dtype=torch.long)

    train_dataset = MultiviewDataset(data_tensor[train_idx], y_train, window_size, mesh_rows, mesh_columns, depth)
    val_dataset = MultiviewDataset(data_tensor[val_idx], y_val, window_size, mesh_rows, mesh_columns, depth)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=multiview_collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=multiview_collate_fn)
    return train_loader, val_loader, train_df2, val_df2

def get_training_parts(model, lr=1e-3, weight_decay=1e-4, max_epochs=100):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max_epochs)
    criterion = torch.nn.CrossEntropyLoss(label_smoothing=0.1)
    return optimizer, scheduler, criterion


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

    plt.title(f"Mean Train/Val Accuracy & Loss (Trial {trail_index})")
    plt.grid(True)
    if path:
        plt.savefig(path, bbox_inches='tight')
    plt.show()
    


def train_model(
    model, train_loader, val_loader, attempt_index, epochs, lr, patience, device,
    all_metrics_df, hyperparams, task_names, fold_idx, min_delta=1e-4, best_model_path=None
):
    model.to(device)
    optimizer, scheduler, criterion = get_training_parts(model, lr=lr)  # Initialize optimizer, scheduler, and criterion

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
def get_hyperparameter_search_space(trial, train_data):
    """
    Define the hyperparameter search space for Optuna.

    Parameters:
        trial (optuna.Trial): The Optuna trial object.
        train_data (torch.Tensor): The training data tensor, used to infer input dimensions.
    Returns:
        dict: A dictionary containing the hyperparameter search space.
    """
    if ModelClass == MultiviewAttention:
        window_size = trial.suggest_categorical('window_size', [2, 4, 8, 16])
        T = train_data.shape[2]
        split_size = T // window_size

        # Find all possible (mesh_rows, mesh_columns) pairs for this split_size
        mesh_row_candidates = [r for r in range(5, min(50, split_size + 1)) if split_size % r == 0]
        if not mesh_row_candidates:
            raise optuna.TrialPruned(f"No valid mesh_rows for split_size={split_size}")

        # Now, mesh_row_candidates is fixed for this trial, so it's safe to suggest
        mesh_rows = mesh_row_candidates[trial.suggest_int('mesh_rows_index', 0, len(mesh_row_candidates) - 1)]
        mesh_columns = split_size // mesh_rows


        hyperparams = {
            "window_size": window_size,
            "conv1_filters": trial.suggest_categorical('conv1_filters', [16, 32, 64]),
            "conv2_filters": trial.suggest_categorical('conv2_filters', [32, 64, 128]),
            "conv3_filters": trial.suggest_categorical('conv3_filters', [64, 128, 256]),
            "conv1_kernel_shape": trial.suggest_categorical('conv1_kernel_shape', [(3, 3), (5, 5), (7, 7)]),
            "conv2_kernel_shape": trial.suggest_categorical('conv2_kernel_shape', [(3, 3), (5, 5), (7, 7)]),
            "conv3_kernel_shape": trial.suggest_categorical('conv3_kernel_shape', [(3, 3), (5, 5), (7, 7)]),
            "padding1": trial.suggest_categorical('padding1', [0, 1, 2, 3]),
            "padding2": trial.suggest_categorical('padding2', [0, 1, 2, 3]),
            "padding3": trial.suggest_categorical('padding3', [0, 1, 2, 3]),
            "conv1_activation": trial.suggest_categorical('conv1_activation', ['relu', 'elu', 'leaky_relu']),
            "conv2_activation": trial.suggest_categorical('conv2_activation', ['relu', 'elu', 'leaky_relu']),
            "conv3_activation": trial.suggest_categorical('conv3_activation', ['relu', 'elu', 'leaky_relu']),
            "dense_nodes": trial.suggest_categorical('dense_nodes', [64, 128]),
            "dense_activation": trial.suggest_categorical('dense_activation', ['relu', 'elu']),
            "depth": trial.suggest_categorical('depth', [train_data.shape[1]]),  # or set appropriately
            "lstm1_cells": trial.suggest_categorical('lstm1_cells', [32, 64, 128]),
            "lstm2_cells": trial.suggest_categorical('lstm2_cells', [32, 64, 128]),
            "dense3_nodes": trial.suggest_categorical('dense3_nodes', [32, 64, 128]),
            "dense3_activation": trial.suggest_categorical('dense3_activation', ['relu', 'elu']),
            "number_classes": len(np.unique(TASKS)),
            "number_channels": train_data.shape[1],  # or set appropriately
            "mesh_rows": mesh_rows,
            "mesh_columns": mesh_columns,
            "split_size": split_size,
            "learning_rate": trial.suggest_float('learning_rate', 5e-5, 5e-2, log=True),
            "batch_size": trial.suggest_categorical('batch_size', [8, 16, 32, 64]),
            "dropout_rate": trial.suggest_float('dropout_rate', 0.1, 0.7, step=0.1),
        }
    else:
        hyperparams = {
        "number_classes": train_data.shape[1],
        "num_temporal_filters": trial.suggest_categorical('num_temporal_filters', [32, 64]),
        "temporal_kernel_size": trial.suggest_categorical('temporal_kernel_size', [32, 64]),
        "dropout_rate": trial.suggest_float('dropout_rate', 0.2, 0.6, step=0.1),
        "activation": trial.suggest_categorical('activation', ['relu', 'gelu']),
        "pool_size": trial.suggest_categorical('pool_size', [2, 4]),
        "number_classes": len(np.unique(TASKS)),
        "conv2d_channels": trial.suggest_categorical("conv2d_channels", ["32_64"]),
        "transformer_d_model": trial.suggest_categorical('transformer_d_model', [128]),
        "transformer_nhead": trial.suggest_categorical('transformer_nhead', [4]),
        "transformer_dim_feedforward": trial.suggest_categorical('transformer_dim_feedforward', [128, 256]),
        "transformer_layers": trial.suggest_int('transformer_layers', 1, 1),
        "fc_hidden": trial.suggest_categorical('fc_hidden', [64, 128]),
        "fc_dropout": trial.suggest_float('fc_dropout', 0.2, 0.6, step=0.1),
        "learning_rate": trial.suggest_float('learning_rate', 1e-4, 5e-3, log=True),
        "batch_size": trial.suggest_categorical('batch_size', [16, 32]),
        "input_time_length": trial.suggest_categorical('input_time_length', [256, 512]),
        "stride": trial.suggest_categorical('stride', [256, 512])
    }
        
    return hyperparams


def objective(trial, all_metrics_df, train_data, train_df, device, n_splits=4, best_model_path=None,
              plot_path=None):
    # Define hyperparameters using trial
    hyperparams = get_hyperparameter_search_space(trial, train_data)

    all_train_accs = []
    all_val_accs = []
    all_train_losses = []
    all_val_losses = []
    val_accs = []

    for fold_idx in range(n_splits):
        if ModelClass == MultiviewAttention:
            train_loader, val_loader, train_df2, val_df2 = get_multiview_dataloaders(
                train_data, train_df,
                fold_index=fold_idx,
                n_splits=n_splits,
                batch_size=hyperparams["batch_size"],
                window_size=hyperparams["window_size"],
                mesh_rows=hyperparams["mesh_rows"],
                mesh_columns=hyperparams["mesh_columns"],
                depth=hyperparams["depth"]
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

        model = ModelClass(**{k: v for k, v in hyperparams.items() if k in ModelClass.__init__.__code__.co_varnames}).to(device)

        trial_index = trial.number + 1

        best_val_acc, updated_metrics_df, train_accs, val_accs_fold, train_losses, val_losses = train_model(
            model, train_loader, val_loader,
            attempt_index=trial_index,
            epochs=25,
            lr=hyperparams["learning_rate"],
            patience=5,
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

    plot_performance(
        mean_train_acc, mean_val_acc, mean_train_loss, mean_val_loss,
        trail_index=trial_index,
        path=plot_path
    )

    return avg_val_acc, all_metrics_df


def evaluate_model_on_testset(model, test_data, test_df, device, batch_size=64, input_time_length=256, stride=128, path=None):
    model.eval()
    if isinstance(model, MultiviewAttention):
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