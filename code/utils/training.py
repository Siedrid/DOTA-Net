from torchmetrics.detection.mean_ap import MeanAveragePrecision
from pathlib import Path
from typing import Callable, Optional, Tuple, Dict, List
import pandas as pd
import numpy as np 
import PIL
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator, TensorEvent
from torch import Tensor
from torchvision import tv_tensors
from torchvision.transforms import v2
import json
import os
import sys
import re


def prepare_map_predictions(predictions: List[Dict[str, torch.Tensor]], threshold: Optional[float] = None) -> List[Dict[str, torch.Tensor]]:
    """
    Convert model predictions to the format required by MeanAveragePrecision.
    Optionally filter predictions with scores below the specified threshold.

    Parameters
    ----------
    predictions : List[Dict[str, torch.Tensor]]
        List of dictionaries from Faster R-CNN, each containing:
        - "boxes": Tensor of predicted bounding boxes (N, 4).
        - "scores": Tensor of predicted scores (N,).
        - "labels": Tensor of predicted labels (N,).
    threshold : float, optional
        Score threshold to filter predictions (None to disable).

    Returns
    -------
    List[Dict[str, torch.Tensor]]
        List of dictionaries in the format required by mAP metric.
    """
    map_preds = []
    for pred in predictions:
        if threshold is not None:
            mask = pred["scores"] > threshold
            filtered_pred = {
                "boxes": pred["boxes"][mask],
                "scores": pred["scores"][mask],
                "labels": pred["labels"][mask]
            }
        else:
            filtered_pred = pred
        map_preds.append(filtered_pred)
    return map_preds

def prepare_map_targets(targets: List[Dict[str, torch.Tensor]]) -> List[Dict[str, torch.Tensor]]:
    """
    Convert ground-truth targets to the format required by MeanAveragePrecision.

    Parameters
    ----------
    targets : List[Dict[str, torch.Tensor]]
        List of dictionaries, each containing:
        - "boxes": Tensor of ground-truth bounding boxes (M, 4).
        - "labels": Tensor of ground-truth labels (M,).

    Returns
    -------
    List[Dict[str, torch.Tensor]]
        List of dictionaries in the format required by mAP metric.
    """
    map_targets = []
    for target in targets:
        map_targets.append({
            "boxes": target["boxes"],
            "labels": target["labels"]
        })
    return map_targets

class EarlyStopping:
    def __init__(self, patience=5, delta=0):
        self.patience = patience
        self.delta = delta
        self.best_score = None
        self.early_stop = False
        self.counter = 0
        self.best_model_state = None

    def __call__(self, val_loss, model):
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
            self.best_model_state = model.state_dict()
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.best_model_state = model.state_dict()
            self.counter = 0
            
    def load_best_model(self, model):
        model.load_state_dict(self.best_model_state)

def get_best_checkpoint_path(file_path: str) -> Optional[str]:
    """
    Extracts the best checkpoint path from the TensorBoard event file.

    This function reads the TensorBoard event file, looks for the scalar summary
    under the tag "Best Checkpoint Path/text_summary", parses the associated tensor
    events, and returns the checkpoint path of the best model.

    Parameters
    ----------
    file_path : str
        The path to the TensorBoard event file that contains the summary information.

    Returns
    -------
    Optional[str]
        The checkpoint path of the best model, or None if no such checkpoint path is found.
    """
    ea = EventAccumulator(file_path)
    ea.Reload()

    for scalar in ea.Tags().get("tensors", []):
        if scalar == "Best Checkpoint Path/text_summary":
            events = ea.Tensors(scalar)
    
    if not events:
        return None
    
    tensor_event_dict = parse_tensor_events(events)
    return tensor_event_dict[-1]["checkpoint_path"]

def parse_tensor_events(tensor_events: List[TensorEvent]) -> List[Dict[str, Optional[float | str]]]:
    """
    Parse a list of tensor events to extract `val_loss` and `checkpoint_path`.

    Parameters
    ----------
    tensor_events : List[TensorEvent]
        A list of TensorEvent objects. Each object is expected to have a `tensor_proto` attribute
        containing a method `SerializeToString()` that returns a serialized byte string.

    Returns
    -------
    List[Dict[str, Optional[float | str]]]
        A list of dictionaries. Each dictionary contains:
        - "val_loss" (float or None): The extracted validation loss value.
        - "checkpoint_path" (str or None): The extracted checkpoint file path.
    """
    parsed_data = []
    for event in tensor_events:
        event_dict = {}
        # Extract the string value (assuming `tensor_proto` contains the relevant string)
        event_string = event.tensor_proto.SerializeToString().decode('utf-8', errors='ignore')
        match = re.search(r'val_loss:([^@]+)', event_string)
        if match:
            val_loss = match.group(1)  # Extract the matched value
            event_dict['val_loss'] = float(val_loss)
        else:
            event_dict['val_loss'] = None
        match = re.search(r'@(.+)', event_string)
        if match:
            path = match.group(1)  # Extract the matched value
            event_dict['checkpoint_path'] = path
        else:
            event_dict['checkpoint_path'] = None
        parsed_data.append(event_dict)
    return parsed_data