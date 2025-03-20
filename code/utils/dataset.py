from pathlib import Path
from typing import Callable, Optional, Tuple, Dict, List
import pandas as pd
import numpy as np 
import PIL
from PIL import Image
Image.MAX_IMAGE_PIXELS = None  # Disables the decompression bomb check
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision.utils import draw_bounding_boxes
from torch import Tensor
from torchvision import tv_tensors
from torchvision.transforms import v2
import torchvision.transforms.functional as F
import json
import os
import sys
import torchvision.transforms.functional as TF
import matplotlib.pyplot as plt

class DOTA_DATASET(Dataset):
    def __init__(self, csv_file: str, root_img_dir: str, tile_size=1024, overlap=200, transform=None):
        self.csv_file = csv_file
        self.annotations = self._read_df()
        self.root_img_dir = root_img_dir
        self.tile_size = tile_size
        self.overlap = overlap
        self.transform = transform
        self.tiles = self._prepare_tiles()

    def _read_df(self):
        """Reads CSV annotations and resets index."""
        return pd.read_csv(self.csv_file).reset_index(drop=True)

    def _prepare_tiles(self):
        """Prepares a list of image tiles with their corresponding bounding boxes."""
        tiles = []
        for idx in range(len(self.annotations)):
            img_name = self.annotations.iloc[idx, 0]
            boxes_string = self.annotations.iloc[idx, 1]
            boxes = [list(map(int, box.split())) for box in boxes_string.split(";")]

            img_path = Path(self.root_img_dir) / img_name
            img = PIL.Image.open(img_path)
            w, h = img.size  # Get original image dimensions

            # Compute tile positions
            stride = self.tile_size - self.overlap
            for y in range(0, h - self.overlap, stride):
                for x in range(0, w - self.overlap, stride):
                    tile_boxes, box_indices = self._adjust_boxes(boxes, x, y, self.tile_size)
                    if tile_boxes:
                        tiles.append((idx, x, y, tile_boxes, box_indices))
        return tiles

    def _adjust_boxes(self, boxes, tile_x, tile_y, tile_size):
        """Adjusts bounding boxes for a given tile and clips if outside tile bounds."""
        tile_boxes = []
        box_indices = []
        for bbox_idx, box in enumerate(boxes):
            x_min, y_min, x_max, y_max = box
            # Shift coordinates relative to the tile
            x_min -= tile_x
            y_min -= tile_y
            x_max -= tile_x
            y_max -= tile_y

            # Clip the bounding box to the tile boundaries (ensure coordinates are within tile size)
            x_min = max(0, x_min)
            y_min = max(0, y_min)
            x_max = min(tile_size, x_max)
            y_max = min(tile_size, y_max)

            # Only include boxes that are inside the tile
            if x_max > x_min and y_max > y_min:
                tile_boxes.append([x_min, y_min, x_max, y_max])
                box_indices.append(bbox_idx)

        return tile_boxes, box_indices

    def __len__(self):
        return len(self.tiles)

    def __getitem__(self, idx):
        img_idx, tile_x, tile_y, tile_boxes, box_indices = self.tiles[idx]
        img_name = Path(self.root_img_dir) / self.annotations.iloc[img_idx, 0]

        # Open and crop image
        full_image = PIL.Image.open(img_name)
        image = TF.crop(full_image, tile_y, tile_x, self.tile_size, self.tile_size)

        # Convert to tv_tensors
        image = tv_tensors.Image(image)
        boxes = tv_tensors.BoundingBoxes(
            tile_boxes, format=tv_tensors.BoundingBoxFormat.XYXY, canvas_size=(self.tile_size, self.tile_size)
        )

        # Get labels
        labels_string = self.annotations.iloc[img_idx, 2]
        labels = [int(cl) for cl in labels_string.split(';')]
        labels = torch.tensor([labels[i] for i in box_indices], dtype=torch.int64)

        # Apply transformations
        sample = {"image": image, "boxes": boxes, "labels": labels}
        if self.transform:
            sample = self.transform(sample)
        target = {"boxes": sample["boxes"], "labels": sample["labels"]}
        return sample["image"], target

## New Dataset Class
class DOTA_DATASET_v2(Dataset):
    def __init__(self, csv_file, root_img_dir, tile_size=1024, overlap=200, transform=None):
        self.annotations = pd.read_csv(csv_file).reset_index(drop=True)
        self.root_img_dir = root_img_dir
        self.tile_size = tile_size
        self.overlap = overlap
        self.transform = transform
        self.tiles = []  # Store precomputed tiles

        # **PRECOMPUTE SLIDING WINDOW TILES**
        for idx in range(len(self.annotations)):
            img_name = self.annotations.iloc[idx, 0]
            boxes_string = self.annotations.iloc[idx, 1]
            labels_string = self.annotations.iloc[idx, 2]

            # Read and convert the full image ONCE
            img_path = Path(self.root_img_dir) / img_name
            img = PIL.Image.open(img_path).convert("RGB")  # Ensure RGB mode
            w, h = img.size

            # Parse bounding boxes
            boxes = [list(map(int, box.split())) for box in boxes_string.split(";")]
            labels = [int(label) for label in labels_string.split(';')]

            stride = tile_size - overlap
            for y in range(0, h - overlap, stride):
                for x in range(0, w - overlap, stride):
                    tile_boxes, box_indices = self._adjust_boxes(boxes, x, y, tile_size)
                    tile_labels = [labels[i] for i in box_indices]

                    if tile_boxes:  # Only store valid tiles with at least one box
                        tile_img = TF.crop(img, y, x, tile_size, tile_size)
                        self.tiles.append((tile_img, tile_boxes, tile_labels))

    def _adjust_boxes(self, boxes, tile_x, tile_y, tile_size):
        """Clips bounding boxes and returns valid ones with indices."""
        tile_boxes, box_indices = [], []
        for i, (x_min, y_min, x_max, y_max) in enumerate(boxes):
            # Shift and clip coordinates
            x_min, y_min, x_max, y_max = x_min - tile_x, y_min - tile_y, x_max - tile_x, y_max - tile_y
            x_min, y_min, x_max, y_max = max(0, x_min), max(0, y_min), min(tile_size, x_max), min(tile_size, y_max)

            # Keep only boxes inside the tile
            if x_max > x_min and y_max > y_min:
                tile_boxes.append([x_min, y_min, x_max, y_max])
                box_indices.append(i)

        return tile_boxes, box_indices

    def __len__(self):
        return len(self.tiles)

    def __getitem__(self, idx):
        image, tile_boxes, tile_labels = self.tiles[idx]

        # Convert to tensors
        image = tv_tensors.Image(image)
        boxes = tv_tensors.BoundingBoxes(tile_boxes, format=tv_tensors.BoundingBoxFormat.XYXY, canvas_size=(self.tile_size, self.tile_size))
        labels = torch.tensor(tile_labels, dtype=torch.int64)

        sample = {"image": image, "boxes": boxes, "labels": labels}

        if self.transform:
            sample = self.transform(sample)

        return sample["image"], {"boxes": sample["boxes"], "labels": sample["labels"]}

def inspect_dataset_sample(
    sample: Tuple[tv_tensors.Image,
    Dict[tv_tensors.BoundingBoxes, torch.Tensor]]
) -> None:
    """
    Inspect a dataset sample by printing its components' types and attributes.
    
    Parameters
    ----------
    sample : Tuple[tv_tensors.Image, Dict[tv_tensors.BoundingBoxes, torch.Tensor]]
        A tuple containing an image tensor and a dictionary with bounding boxes and labels.
    """
    img, target = sample
    def print_inspection(title: str, data: Dict[str, Any]) -> None:
        print(title)
        for key, value in data.items():
            print(f"{key} = {value}")
        print("")
    print_inspection("Inspect img:", {
        "type(img)": type(img),
        "img.shape": img.shape,
        "img.dtype": img.dtype
    })
    print_inspection("Inspect target:", {
        "type(target)": type(target),
        "target.keys()": target.keys()
    })
    print_inspection("Inspect target['boxes']:", {
        "type(target['boxes'])": type(target['boxes']),
        "target['boxes'].dtype": target['boxes'].dtype,
        "target['boxes'].canvas_size": target['boxes'].canvas_size,
        "target['boxes'].format": target['boxes'].format
    })
    print_inspection("Inspect target['labels']:", {
        "type(target['labels'])": type(target['labels']),
        "target['labels'].dtype": target['labels'].dtype
    })


def plot_image_with_boxes(
    image: torch.Tensor,
    boxes: torch.Tensor,
    denormalize_values: Optional[Tuple[List[float], List[float]]] = None,
    labels: Optional[List[str]] = None
) -> None:
    """
    Plot the image with bounding boxes using torchvision utilities.

    Parameters
    ----------
    image : torch.Tensor
        The image tensor in CHW format.
    boxes : torch.Tensor
        A tensor containing bounding boxes with shape (N, 4), where each row is (x_min, y_min, x_max, y_max).
    denormalize_values : Optional[Tuple[List[float], List[float]]], optional
        Mean and standard deviation values for denormalization, by default None.
    labels : Optional[List[str]], optional
        Labels for each bounding box, by default None.
    """
    img = image.clone()

    if denormalize_values is not None:
        mean, std = denormalize_values
        transform = transforms.Normalize(mean=[-m / s for m, s in zip(mean, std)],
                                         std=[1 / s for s in std])
        img = transform(img)
        img = (img * 255).byte()

    img_with_boxes = draw_bounding_boxes(img, boxes, labels=labels, colors="red", width=2)
    img_pil = F.to_pil_image(img_with_boxes)

    plt.imshow(img_pil)
    plt.axis("off")
    plt.show()