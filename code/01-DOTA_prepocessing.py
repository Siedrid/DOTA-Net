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

# Potential Problems:
# from oriented to horizontal bbox
# different img size, GSD
# Potential architecture: YOLO
# was passiert mit bildern ohne eine klasse drin


DOTA_SET = 'dota' # possible values: dota-subset, dota
SPLIT = 'train' # possible values: train, val, test-dev

DATA_ROOT = Path('/dss/dsstbyfs02/pn49ci/pn49ci-dss-0022/data/')
DOTA_ROOT = DATA_ROOT / DOTA_SET

META_FILE = DOTA_ROOT / 'meta.json'
LABELS_DIR = DOTA_ROOT / SPLIT / 'ann'
IMGS_DIR = DOTA_ROOT / SPLIT / 'img'
csv_file = DOTA_ROOT / f'{SPLIT}_split.csv'
df = pd.read_csv(csv_file)
df.columns
df.bbox[0]
len(df)

df['class']


with open(META_FILE) as f:
    meta = json.load(f)


class_lookup = {
    cls["hotkey"]: {"id": int(1 + i), "title": cls.get("title", "No title")}
    for i, cls in enumerate(meta["classes"]) if "hotkey" in cls
    }

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
        target = {"boxes": sample["boxes"], "labels": sample["labels"]
        }
        return sample["image"], target


dataset = DOTA_DATASET(csv_file, IMGS_DIR)
loader = DataLoader(dataset, batch_size = 16)
nimages = 0
mean = 0.
std = 0.
for batch, _ in loader:
    # Rearrange batch to be the shape of [B, C, W * H]
    batch = batch.view(batch.size(0), batch.size(1), -1)
    # Update total number of images
    nimages += batch.size(0)
    # Compute mean and std here
    mean += batch.mean(2).sum(0) 
    std += batch.std(2).sum(0)

# Final step
mean /= nimages
std /= nimages

print(mean)
print(std)


# Transform Function
def transforms() -> v2.Compose:
        """

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
                v2.SanitizeBoundingBoxes(), # if bounding box was moved out of the image
                v2.ToDtype(torch.float32, scale=True),
                v2.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]) # better would be the mean and std of the actual wheat head images
            ]
        )

# attribute canvas_size for padding
# exclude classes like the airport, which are too big


image, target = dataset[0]
image.shape
image = image.float() / 255.0 
mn = torch.mean(image[0,:,:])
labs = torch.ones((4,), dtype=torch.int64)
target