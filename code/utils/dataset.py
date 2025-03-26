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
from shapely.geometry import box

## New Dataset Class
import torch
import torchvision.transforms.functional as TF
import pandas as pd
import PIL.Image
from pathlib import Path
from torch.utils.data import Dataset
import torchvision.tv_tensors as tv_tensors

# add progress for dataset loading
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
            boxes_string = str(self.annotations.iloc[idx, 1])
            labels_string = str(self.annotations.iloc[idx, 2])

            # Read and convert the full image ONCE
            img_path = Path(self.root_img_dir) / img_name
            img = PIL.Image.open(img_path) 

            # Pad the image if it's smaller than tile_size
            img, pad_left, pad_top = self._pad_image(img)

            w, h = img.size  # Updated size after padding

            #print(boxes_string)
            # Parse bounding boxes
            boxes = [list(map(int, box.split())) for box in boxes_string.split(";") if box != 'nan']
            labels = [int(label) for label in labels_string.split(';') if label.strip().isdigit()]

            # Adjust bounding boxes for padding
            boxes = [[x_min + pad_left, y_min + pad_top, x_max + pad_left, y_max + pad_top] for x_min, y_min, x_max, y_max in boxes]

            stride = tile_size - overlap
            for y in range(0, h - overlap, stride):
                for x in range(0, w - overlap, stride):
                    tile_boxes, box_indices = self._adjust_boxes(boxes, x, y, tile_size)
                    tile_labels = [labels[i] for i in box_indices]

                    if tile_boxes:  # Only store valid tiles with at least one box
                        tile_img = TF.crop(img, y, x, tile_size, tile_size)
                        self.tiles.append((tile_img, tile_boxes, tile_labels))

    def _pad_image(self, img):
        """Pads images smaller than tile_size to tile_size x tile_size with zeros."""
        w, h = img.size
        pad_left = max(0, (self.tile_size - w) // 2)  # Center horizontally
        pad_top = max(0, (self.tile_size - h) // 2)  # Center vertically
        pad_right = max(0, self.tile_size - (w + pad_left))  # Ensure full padding
        pad_bottom = max(0, self.tile_size - (h + pad_top))

        img = TF.pad(img, (pad_left, pad_top, pad_right, pad_bottom), fill=0)  # Zero-padding (black)
        return img, pad_left, pad_top  # Return image and padding values

    def _adjust_boxes(self, boxes, tile_x, tile_y, tile_size, iou_threshold=0.7):
        """Clips bounding boxes and returns valid ones with indices."""
        tile_boxes, box_indices = [], []
        tile_poly = box(tile_x, tile_y, tile_x + tile_size, tile_y + tile_size)

        for i, (x_min, y_min, x_max, y_max) in enumerate(boxes):
            box_poly = box(x_min, y_min, x_max, y_max)
            inter_poly, half_iou = self._calc_half_iou(box_poly, tile_poly)
            
            if half_iou >= iou_threshold:
                x_min, y_min, x_max, y_max = map(int, inter_poly.bounds)
                x_min, y_min, x_max, y_max = x_min - tile_x, y_min - tile_y, x_max - tile_x, y_max - tile_y
                x_min, y_min, x_max, y_max = max(0, x_min), max(0, y_min), min(tile_size, x_max), min(tile_size, y_max)
                if x_max > x_min and y_max > y_min:
                    tile_boxes.append([x_min, y_min, x_max, y_max])
                    box_indices.append(i)

        return tile_boxes, box_indices

    def _calc_half_iou(self, poly1, poly2):
        """
        It is not the IoU on usual, the IoU is the value of intersection over poly1
        https://github.com/dingjiansw101/AerialDetection/blob/master/DOTA_devkit/ImgSplit_multi_process.py#L163
        """
        inter_poly = poly1.intersection(poly2)
        inter_area = inter_poly.area
        poly1_area = poly1.area
        half_iou = inter_area / poly1_area if poly1_area > 0 else 0
        return inter_poly, half_iou

    def __len__(self):
        return len(self.tiles)

    def __getitem__(self, idx):
        image, tile_boxes, tile_labels = self.tiles[idx]

        # Convert to tensors
        image = tv_tensors.Image(image)
        boxes = tv_tensors.BoundingBoxes(
            tile_boxes, 
            format=tv_tensors.BoundingBoxFormat.XYXY, 
            canvas_size=image.shape[-2:])
        
        labels = torch.tensor(tile_labels, dtype=torch.int64)

        sample = {"image": image, "boxes": boxes, "labels": labels}

        if self.transform:
            sample = self.transform(sample)

        return sample["image"], {"boxes": sample["boxes"], "labels": sample["labels"]}

# Dataset class for preprocessed DOTA images
class DOTA_preprocessed(Dataset):
    def __init__(self, csv_file, root_img_dir, transform=None):
        self.csv_file = csv_file
        self.annotations = self._exclude_no_box_samples()
        self.root_img_dir = root_img_dir
        self.transform = transform

    def _exclude_no_box_samples(self) -> pd.DataFrame:
        """Exclude samples where BoxesString is 'no_box'.

        Returns
        -------
        pd.DataFrame
            The annotations data frame without the samples where BoxesString is 'no_box'.
        """
        annotations = pd.read_csv(self.csv_file)
        annotations = annotations.dropna()
        return annotations.reset_index(drop=True)

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        img_name = self.annotations.iloc[idx, 0]
        boxes_string = str(self.annotations.iloc[idx, 2])
        labels_string = str(self.annotations.iloc[idx, 3])

        img_path = Path(self.root_img_dir) / img_name
        img = tv_tensors.Image(PIL.Image.open(img_path))
        
        # check this
        boxes = [list(map(int, box.split())) for box in boxes_string.split(";") if box != 'nan']
        labels = [int(label) for label in labels_string.split(';') if label.strip().isdigit()]
        labels = torch.tensor([labels[i] for i in range(len(boxes))], dtype=torch.int64)

        boxes = tv_tensors.BoundingBoxes(
            boxes,
            format=tv_tensors.BoundingBoxFormat.XYXY, # specific format with xmin ymin etc.
            canvas_size=img.shape[-2:]
        ) 

        sample = {"image": img, "boxes": boxes, "labels": labels}
        if self.transform:
            sample = self.transform(sample)
        # Create the target dictionary
        target = {"boxes": sample["boxes"], "labels": sample["labels"]
        }
        return sample["image"], target

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