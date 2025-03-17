#%%
from pathlib import Path
from typing import Callable, Optional, Tuple, Dict, List
import pandas as pd
import numpy as np 
import PIL
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

# Dataset
class DOTA_DATASET(Dataset):
    def __init__(self, csv_file: str, root_img_dir: str, transform: Optional[Callable] = None) -> None:
        self.csv_file = csv_file
        self.annotations = self._read_df()
        self.root_img_dir = root_img_dir
        self.transform = transform

    def _read_df(self) -> pd.DataFrame:
        annotations = pd.read_csv(self.csv_file)
        # add filters, maybe for difficulty
        # filter for image size
        return annotations.reset_index(drop=True)

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx: int):
        # Get the image file name from the annotations DataFrame
        img_name = Path(self.root_img_dir) / self.annotations.iloc[idx, 0]
        # Open the image using Pillow (PIL) and create a torchvision TVTensor Image
        image = tv_tensors.Image(PIL.Image.open(img_name))

        # Extract bounding box coordinates from the 'BoxesString' column
        boxes_string = self.annotations.iloc[idx, 1]
        boxes = []
        for box in boxes_string.split(";"): # Split the string into individual box coordinates
            x_min, y_min, x_max, y_max = map(int, box.split())  # Convert coordinates to integers
            boxes.append([x_min, y_min, x_max, y_max]) # Add the box coordinates to the list

        # Create a torchvision TVTensor BoundingBoxes object
        boxes = tv_tensors.BoundingBoxes(
            boxes,
            format=tv_tensors.BoundingBoxFormat.XYXY,
            canvas_size=image.shape[-2:] # check this
        )

        # Create a PyTorch tensor for the labels 
        labels_string = self.annotations.iloc[idx, 2]
        labels = []
        for cl in labels_string.split(';'):
            labels.append(int(cl)) # dtype = torch.int64

        # to tensor
        labels_tensor = torch.tensor(labels, dtype=torch.int64)

        sample = {"image": image, "boxes": boxes, "labels": labels_tensor}
        if self.transform:
            sample = self.transform(sample)
        # Create the target dictionary
        target = {"boxes": sample["boxes"], "labels": sample["labels"]}
        return sample["image"], target

def get_max_dimensions(csv_file):

    df = pd.read_csv(csv_file)
    max_h = max(df.height)
    max_w = max(df.width)

    return max_w, max_h

max_w, max_h = get_max_dimensions(csv_file)
target_image_size = (29200, 29200)  # as an example

## Padding 
import torchvision.transforms.functional as F
from torchvision import transforms
import numbers
import torch
import torch.nn.functional as F
from torchvision import tv_tensors

class CustomPadding(torch.nn.Module):
    def __init__(self, target_size=2000):
        super().__init__()
        self.target_size = target_size  # Target image size (H, W)

    def get_padding(self, image):
        """Computes padding required to reach the target size."""
        _, h, w = image.shape  # Assuming image format is (C, H, W)

        pad_left = (self.target_size - w) // 2
        pad_right = self.target_size - w - pad_left
        pad_top = (self.target_size - h) // 2
        pad_bottom = self.target_size - h - pad_top

        return pad_left, pad_right, pad_top, pad_bottom

    def forward(self, sample):
        """Applies padding to both image and bounding boxes."""
        image = sample['image']
        boxes = sample['boxes']

        # Compute padding values
        pad_left, pad_right, pad_top, pad_bottom = self.get_padding(image)

        # Pad image (F.pad expects (left, right, top, bottom))
        padded_image = F.pad(image, (pad_left, pad_right, pad_top, pad_bottom), value=0)

        # Adjust bounding boxes
        adjusted_boxes = torch.stack([
            boxes[:, 0] + pad_left,  # x_min
            boxes[:, 1] + pad_top,   # y_min
            boxes[:, 2] + pad_left,  # x_max
            boxes[:, 3] + pad_top    # y_max
        ], dim=1)

        # Update sample with padded image and adjusted bounding boxes
        sample['image'] = tv_tensors.Image(padded_image)
        sample['boxes'] = tv_tensors.BoundingBoxes(adjusted_boxes, format="XYXY", canvas_size=(self.target_size, self.target_size))

        return sample

transform = v2.Compose([
    CustomPadding()
])
dataset = DOTA_DATASET(csv_file, IMGS_DIR, transform=transform)
image, target = dataset[0]
print(image.shape)
utils.plot_image_with_boxes(image, target['boxes'])
'''
#
dataset = DOTA_DATASET(csv_file, IMGS_DIR)
image, target = dataset[0]
print(image.shape)
loader = DataLoader(dataset, batch_size = 16)

for batch, sample in loader:
    print(batch.size)
'''

#%%
loader = DataLoader(dataset, batch_size = 16)

for batch, sample in loader:
    print(batch.size)

# Transform Function
transfrom = transforms.Compose
# attribute canvas_size for padding
# exclude classes like the airport, which are too big


image, target = dataset[0]
image.shape
image = image.float() / 255.0 
mn = torch.mean(image[0,:,:])
labs = torch.ones((4,), dtype=torch.int64)
target