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
from Model import MEGClassifier, TinyMLP
from sklearn.model_selection import StratifiedShuffleSplit

# Gloval variables
debug_mode = False
TASKS = ["resting", "math", "memory", "motor"]
METRICS = ["accuracy", "F1", "recall", "precision"]

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
    data_tensor, df, batch_size=64, input_time_length=256, stride=128, shuffle=False
):
    """
    Create a DataLoader for EEG/MEG data using sliding windows.

    Parameters
    ----------
    data_tensor : torch.Tensor
        EEG data of shape (N, C, T).
    df : pd.DataFrame
        Metadata DataFrame with a 'task_type_cat' column.
    batch_size : int, optional
        Batch size for DataLoader (default: 64).
    input_time_length : int, optional
        Length of each sliding window (default: 256).
    stride : int, optional
        Stride for sliding window extraction (default: 128).
    shuffle : bool, optional
        Whether to shuffle the dataset (default: False).

    Returns
    -------
    loader : DataLoader
        DataLoader for the dataset.
    X : torch.Tensor
        Windows.
    y : torch.Tensor
        Labels.
    """
    X, sample_indices = extract_sliding_windows(data_tensor, window_size=input_time_length, stride=stride)
    X = X.float()
    y_full = df['task_type_cat'].values
    y = torch.tensor(y_full[sample_indices], dtype=torch.long)
    loader = DataLoader(TensorDataset(X, y), batch_size=batch_size, shuffle=shuffle)
    return loader, X, y


def get_dataloaders(
        data_tensor, df, batch_size=64, val_split=0.2, input_time_length=256, stride=128
    ):
    """
    Split at the sample level, then extract windows for each split.
    """
    # Stratified split at the sample level
    y = df['task_type_cat'].values
    sss = StratifiedShuffleSplit(n_splits=1, test_size=val_split, random_state=42)
    train_idx, val_idx = next(sss.split(np.zeros(len(df)), y))

    # Split data and df
    train_data, val_data = data_tensor[train_idx], data_tensor[val_idx]
    train_df, val_df = df.iloc[train_idx].reset_index(drop=True), df.iloc[val_idx].reset_index(drop=True)

    print("train task types counts:")
    print(np.bincount(train_df['task_type'].astype('category').cat.codes))
    print("val task types counts:")
    print(np.bincount(val_df['task_type'].astype('category').cat.codes))

    # Extract windows for each split
    train_X, train_sample_indices = extract_sliding_windows(train_data, window_size=input_time_length, stride=stride)
    val_X, val_sample_indices = extract_sliding_windows(val_data, window_size=input_time_length, stride=stride)

    train_y_full = train_df['task_type_cat'].values
    val_y_full = val_df['task_type_cat'].values

    train_y = torch.tensor(train_y_full[train_sample_indices], dtype=torch.long)
    val_y = torch.tensor(val_y_full[val_sample_indices], dtype=torch.long)

    train_loader = DataLoader(TensorDataset(train_X, train_y), batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(TensorDataset(val_X, val_y), batch_size=batch_size, shuffle=False)
    return train_loader, val_loader, train_X, train_y, val_X, val_y

## tainging and logging related functions
def create_metrics_df():
    """
    Create a DataFrame to log training metrics.
    """

    # Define the metrics and tasks
    metrics = ["accuracy", "F1", "recall", "precision"]
    tasks = ["overall", "resting", "math", "memory", "motor"]
    hyperparameters = [
        "input_channels",
        "num_temporal_filters",
        "temporal_kernel_size",
        "dropout_rate",
        "activation",
        "pool_size",
        "num_classes",
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

    # Generate columns for train, val, and test metrics
    columns = ["trial_id", "epoch"]
    for split in ["train", "val", "test"]:
        for task in tasks:
            for metric in metrics:
                columns.append(f"{split}_{task}_{metric}")

    # Add hyperparameter columns
    columns.extend(hyperparameters)

    # Create the DataFrame
    all_metrics_df = pd.DataFrame(columns=columns)

    # Example: Print the generated columns
    if debug_mode:
        print("Generated columns:")
        print(all_metrics_df.columns)

    return all_metrics_df

def train_model(
    model, train_loader, val_loader, attempt_index, epochs, lr, patience, device,
    all_metrics_df, hyperparams, task_names
):
    import matplotlib.pyplot as plt

    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', patience=5, factor=0.5)
    criterion = torch.nn.CrossEntropyLoss(label_smoothing=0.1)

    best_val_acc = 0.0
    best_model_state = model.state_dict()
    no_improvement_epochs = 0

    # For plotting
    train_accs, val_accs = [], []
    train_f1s, val_f1s = [], []

    # Progress bar for epochs
    epochs_pbar = tqdm(range(epochs), desc=f"Trial: {attempt_index}", unit="epoch")

    for epoch in epochs_pbar:
        # --- Training ---
        model.train()
        train_preds, train_targets = [], []
        for x_batch, y_batch in train_loader:
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            outputs = model(x_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()
            train_preds.extend(outputs.argmax(dim=1).detach().cpu().numpy())
            train_targets.extend(y_batch.detach().cpu().numpy())
        train_preds = np.array(train_preds)
        train_targets = np.array(train_targets)

        # --- Validation ---
        model.eval()
        val_preds, val_targets = [], []
        with torch.no_grad():
            for x_batch, y_batch in val_loader:
                x_batch, y_batch = x_batch.to(device), y_batch.to(device)
                outputs = model(x_batch)
                val_preds.extend(outputs.argmax(dim=1).detach().cpu().numpy())
                val_targets.extend(y_batch.detach().cpu().numpy())
        val_preds = np.array(val_preds)
        val_targets = np.array(val_targets)

        # --- Metrics ---
        row = {"trial_id": attempt_index, "epoch": epoch + 1}
        # Compute metrics for train and val
        train_metrics = model.evaluate(train_targets, train_preds)
        val_metrics = model.evaluate(val_targets, val_preds)

        # Store for plotting
        train_accs.append(train_metrics["overall_accuracy"])
        val_accs.append(val_metrics["overall_accuracy"])
        train_f1s.append(train_metrics["overall_F1"])
        val_f1s.append(val_metrics["overall_F1"])

        # Fill DataFrame row for all columns
        for split, metrics_dict in zip(["train", "val"], [train_metrics, val_metrics]):
            for key, value in metrics_dict.items():
                col = f"{split}_{key}"
                row[col] = value
        # (test columns left empty here)
        for task in task_names:
            for metric in METRICS:
                row[f"test_{task}_{metric}"] = np.nan
        # Add hyperparameters
        for k, v in hyperparams.items():
            row[k] = v
        # Append to DataFrame
        all_metrics_df = pd.concat([all_metrics_df, pd.DataFrame([row])], ignore_index=True)

        # Early stopping
        val_acc = val_metrics["overall_accuracy"]
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model_state = model.state_dict()
            no_improvement_epochs = 0
        else:
            no_improvement_epochs += 1
        if no_improvement_epochs >= patience:
            print(f"Early stopping triggered after {epoch + 1} epochs.")
            break
        scheduler.step(val_acc)

        # Update progress bar
        epochs_pbar.set_postfix(
            train_accuracy=train_metrics["overall_accuracy"],
            val_accuracy=val_metrics["overall_accuracy"],
            train_F1=train_metrics["overall_F1"],
            val_F1=val_metrics["overall_F1"],
            train_recall=train_metrics["overall_recall"],
            val_recall=val_metrics["overall_recall"],
            train_precision=train_metrics["overall_precision"],
            val_precision=val_metrics["overall_precision"]
        )

    model.load_state_dict(best_model_state)

    # --- Plotting ---
    plt.figure(figsize=(10, 5))
    plt.plot(train_accs, label="Train Accuracy")
    plt.plot(val_accs, label="Val Accuracy")
    plt.plot(train_f1s, label="Train F1", linestyle="--")
    plt.plot(val_f1s, label="Val F1", linestyle="--")
    plt.xlabel("Epoch")
    plt.ylabel("Score")
    plt.title(f"Training/Validation Accuracy & F1 (Trial {attempt_index})")
    plt.legend()
    plt.grid(True)
    plt.show()

    return best_val_acc, all_metrics_df

# --- Optuna workflow ---
def objective(trial, all_metrics_df, train_data, train_df, device):
    # Define all hyperparameters
    hyperparams = {
        "input_channels": train_data.shape[1],
        "num_temporal_filters": trial.suggest_categorical('num_temporal_filters', [8, 16]),  # reduce max
        "temporal_kernel_size": trial.suggest_categorical('temporal_kernel_size', [32, 64]),  # reduce max
        "dropout_rate": trial.suggest_float('dropout_rate', 0.5, 0.8, step=0.1),  # increase min/max
        "activation": trial.suggest_categorical('activation', ['relu', 'gelu']),
        "pool_size": trial.suggest_categorical('pool_size', [2, 4]),
        "num_classes": len(np.unique(train_df['task_type'])),
        "conv2d_channels": trial.suggest_categorical("conv2d_channels", ["64_128"]),  # only smaller
        "transformer_d_model": trial.suggest_categorical('transformer_d_model', [128]),  # only smaller
        "transformer_nhead": trial.suggest_categorical('transformer_nhead', [4]),  # only smaller
        "transformer_dim_feedforward": trial.suggest_categorical('transformer_dim_feedforward', [128, 256]),  # reduce max
        "transformer_layers": trial.suggest_int('transformer_layers', 1, 1),  # only 1 layer
        "fc_hidden": trial.suggest_categorical('fc_hidden', [64, 128]),  # reduce max
        "fc_dropout": trial.suggest_float('fc_dropout', 0.5, 0.8, step=0.1),  # increase min/max
        "learning_rate": trial.suggest_float('learning_rate', 1e-4, 5e-3, log=True),  # keep as is or reduce max
        "batch_size": trial.suggest_categorical('batch_size', [32, 64]),  # reduce max
        "input_time_length": trial.suggest_categorical('input_time_length', [256, 512]),
        "stride": trial.suggest_categorical('stride', [256, 512])  # large stride = less overlap
    }

    # When instantiating the model, unpack the dictionary:
    # model = MEGClassifier(**{k: v for k, v in hyperparams.items() if k in MEGClassifier.__init__.__code__.co_varnames}).to(device)
    model = TinyMLP(**{k: v for k, v in hyperparams.items() if k in TinyMLP.__init__.__code__.co_varnames}).to(device)

    # DataLoaders
    train_loader, val_loader, train_X, train_y, val_X, val_y = get_dataloaders(
        train_data, train_df, batch_size=hyperparams["batch_size"],
        input_time_length=hyperparams["input_time_length"], stride=hyperparams["stride"]
    )

    # Train
    trial_index = trial.number + 1
    best_val_acc, updated_metrics_df = train_model(
        model, train_loader, val_loader, attempt_index=trial_index, epochs=25, lr=hyperparams["learning_rate"],
        patience=10, device=device, all_metrics_df=all_metrics_df, hyperparams=hyperparams, task_names=TASKS
    )
    # Update global metrics DataFrame
    all_metrics_df = updated_metrics_df
    return best_val_acc, all_metrics_df #  return negative accuracy for minimization and to update the DataFrame


def evaluate_model_on_testset(model, test_data, test_df, device, batch_size=64, input_time_length=256, stride=128):
    """
    Evaluates the model on the test set and returns:
    - overall accuracy, F1, recall, precision
    - per-task metrics as a dict: {task_name: {metric: value, ...}, ...}
    Assumes test_df['task_type'] contains the true labels (categorical codes in 'task_type_cat').
    """
    model.eval()

    # get test loader
    test_loader, test_X, test_y = get_dataloader(data_tensor=test_data, df=test_df,
                                        batch_size=batch_size, input_time_length=input_time_length,
                                          stride=stride, shuffle=False)
    # get test predictions
    y_pred_test = model.predict(test_loader, device=device)
    # evaluate the model's predictions on the test set
    test_metrics = model.evaluate(y_pred=y_pred_test, y_true=test_y)

    return test_metrics