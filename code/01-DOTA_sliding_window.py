#%%
from pathlib import Path
from typing import Callable, Optional, Tuple, Dict, List
import pandas as pd
import numpy as np 
import PIL
from PIL import Image
Image.MAX_IMAGE_PIXELS = None  # Disables the decompression bomb check
import torch
from torch.utils.data import Dataset, DataLoader
from torch import Tensor
from torchvision import tv_tensors
from torchvision.transforms import v2
import json
import os
import sys
sys.path.append('/dss/dsshome1/0A/di38tac/DOTA-Net/code')
import utils.dataset as utils
# Potential Problems:
# from oriented to horizontal bbox
# different img size, GSD
# Potential architecture: YOLO
# was passiert mit bildern ohne eine klasse drin


DOTA_SET = 'dota-subset' # possible values: dota-subset, dota
SPLIT = 'train' # possible values: train, val, test-dev

DATA_ROOT = Path('/dss/dsstbyfs02/pn49ci/pn49ci-dss-0022/data/')
DOTA_ROOT = DATA_ROOT / DOTA_SET

META_FILE = DOTA_ROOT / 'meta.json'
LABELS_DIR = DOTA_ROOT / SPLIT / 'ann'
IMGS_DIR = DOTA_ROOT / SPLIT / 'img'
CSV_DIR = Path(f'/dss/dsshome1/0A/di38tac/DOTA-Net/Data/csv-files/{DOTA_SET}')
csv_file = CSV_DIR / f'{SPLIT}_split.csv'
df = pd.read_csv(csv_file)

import torch
import pandas as pd
import PIL.Image
from pathlib import Path
from torchvision import tv_tensors
from torch.utils.data import Dataset
import torchvision.transforms.functional as TF

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


# Transformations
def transforms() -> v2.Compose:
        """
        Define the transformations to be applied to the dataset samples.

        In this case, we apply the following transformations:
        - RandomVerticalFlip, which applies a random vertical flip with a probability of 0.5
        - RandomHorizontalFlip, which applies a random horizontal flip with a probability of 0.5
        - SanitizeBoundingBoxes, which ensures that bounding boxes are within the image boundaries
        - ToDtype, which converts the image to float32 and scales it to [0, 1]
        - Normalize, which normalizes the image pixel values to a specific range with the ImageNet mean and standard deviation

        Returns
        -------
        v2.Compose
            A composition of the transformations to be applied to the dataset samples.
        """
        return v2.Compose(
            [
                # Order Matters!!
                v2.RandomVerticalFlip(p=0.5),
                v2.RandomHorizontalFlip(p=0.5),
                v2.SanitizeBoundingBoxes() # if bounding box was moved out of the image
            ]
        )

dataset = DOTA_DATASET(csv_file=csv_file, root_img_dir=IMGS_DIR, tile_size=1024, overlap=200, transform=transforms())
image, target = dataset[2]  # Get the first tile
print(image.shape)  # Should be (3, 1024, 1024)
print(target["boxes"])  # Bounding boxes adjusted for the tile
print(target['labels'])
utils.plot_image_with_boxes(image, target['boxes'])

def collate_fn(batch: List[Tuple[Tensor, Dict[str, Tensor]]]) -> Tuple[List[Tensor], List[Dict[str, Tensor]]]:
    """
    Custom collate function to correctly batch the provided tensors by the dataset.

    Parameters
    ----------
    batch : List[Tuple[Tensor, Dict[str, Tensor]]]
        A list of tuples where each tuple contains an image tensor and its corresponding target dictionary.

    Returns
    -------
    Tuple[List[Tensor], List[Dict[str, Tensor]]]
        A tuple containing two lists - one for images and one for target dictionaries.
    """
    return tuple(zip(*batch))

data_loader = DataLoader(
    dataset,
    batch_size=2,
    collate_fn=collate_fn,
)
#%%