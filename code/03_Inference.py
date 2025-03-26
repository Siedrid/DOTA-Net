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

import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor, FasterRCNN_ResNet50_FPN_Weights
from torchinfo import summary
from torch.optim import SGD
from torch.utils.tensorboard import SummaryWriter
from torchmetrics.detection.mean_ap import MeanAveragePrecision

import sys
sys.path.append('/dss/dsshome1/0A/di38tac/DOTA-Net/code')
from utils.dataset import DOTA_DATASET_v2
from utils.training import prepare_map_predictions, prepare_map_targets
from utils.training import get_best_checkpoint_path

DOTA_SET = 'dota-subset' # possible values: dota-subset, dota
SPLIT = 'test-dev' # possible values: train, val, test-dev

DATA_ROOT = Path('/dss/dsstbyfs02/pn49ci/pn49ci-dss-0022/data/')
DOTA_ROOT = DATA_ROOT / DOTA_SET

META_FILE = DOTA_ROOT / 'meta.json'
LABELS_DIR = DOTA_ROOT / SPLIT / 'ann'
IMGS_DIR = DOTA_ROOT / SPLIT / 'img'

ROOT = Path("/dss/dsstbyfs02/pn49ci/pn49ci-dss-0022")
USER = "di38tac"
USER_PATH = ROOT / f"users/{USER}"

model_name = 'FasterRCNN'

## or upload model to github
best_checkpoint_path = get_best_checkpoint_path("path")
model = torchvision.models.detection.fasterrcnn_resnet50_fpn()
num_classes = 19
in_features = model.roi_heads.box_predictor.cls_score.in_features

model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

checkpoint = torch.load(best_checkpoint_path, weights_only=False)
model.load_state_dict(checkpoint['model_state_dict'])

## new inference dataset class needed
def inference_transforms() -> v2.Compose:
    """
    Define the transformations to be applied to the dataset samples.

    In this case, we apply the following transformations:
    - ToDtype, which converts the image to float32 and scales it to [0, 1]

    Returns
    -------
    v2.Compose
        A composition of the transformations to be applied to the dataset samples.
    """
    return v2.Compose(
        [
            v2.ToDtype(torch.float32, scale=True)
        ]
    )

# Pipeline:
# Preprocess the images 
from utils.preprocess_dota import preprocess_dota_dataset

print("Preprocessing dataset...")
out_split_imgs = USER_PATH / "DATA" / "SlidingWindow" / DOTA_SET / SPLIT
os.makedirs(out_split_imgs, exist_ok=True)

# do this also in batches
# or change to a simple sliding window function, which is part of all the other functions

preprocess_dota_dataset(
    csv_file=LABELS_DIR / f"{SPLIT}.csv",
    root_img_dir=IMGS_DIR,
    output_dir=out_split_imgs,
    tile_size=1024,
    overlap=200,
    boxes=None
)

print(f"Preprocessing done. Image Tiles saved to {out_split_imgs}.")
# Create the dataset class

class DOTA_Inference(Dataset):
    def __init__(self, csv_file, root_img_dir, transform=None):
        self.annotations = self._read_df()
        self.root_img_dir = root_img_dir
        self.transform = transform

    def _read_df(self):
        return pd.read_csv(self.csv_file).reset_index(drop=True)

    def __len__(self):
        return len(self.annotations)
    
    def __getitem__(self, index):
        img_name = Path(self.root_img_dir) / self.annotations.iloc[index, 0]
        image = tv_tensors.Image(PIL.Image.open(img_name))

        # get the image identifier to pass it to the results dict
        image_id = self.annotations.iloc[index, 0]
        sample = {"image": image, "image_id": image_id}

        if self.transform:
            sample = self.transform(sample)
        return sample['image'], sample['image_id']
    
# Load the model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

# make predictions
def predict(dataloader, model):
    results = []
    model.eval()
    model.to(device)
    with torch.no_grad():
        for images, image_ids in dataloader:
            images = [image.to(device) for image in images]
            image_id = [image_id for image_id in image_ids]

            predictions = model(images)
            for i, prediction in enumerate(predictions):
                boxes = prediction["boxes"].cpu().numpy()
                labels = prediction["labels"].cpu().numpy()
                scores = prediction["scores"].cpu().numpy()
                results.append(
                    {
                        "image_id": image_id[i],
                        "boxes": boxes,
                        "labels": labels,
                        "scores": scores,
                    }
                )
    return results

inference_dataset = DOTA_Inference(
    csv_file=out_split_imgs / "ann" / "annotations.csv",
    root_img_dir=out_split_imgs / "img",
    transform=inference_transforms()
)

inference_data_loader = DataLoader(
    inference_dataset,
    batch_size=124,
    shuffle=False,
    num_workers=4,
    prefetch_factor=2,
    drop_last=False
)

predictions = predict(inference_data_loader, model)

# write predictions to csv
# convert boxes and labels to strings
predicted = []
for prediction in predictions:
    boxes = prediction["boxes"]
    labels = prediction["labels"]
    scores = prediction["scores"]

    new_box_string = ";".join([" ".join(map(str, box)) for box in boxes])
    new_label_string = ";".join(map(str, labels))

    annotations = [prediction["image_id"], new_box_string, new_label_string, scores]
    
    predicted.append(annotations)

prediction_df = pd.DataFrame(predicted, columns=["image_id", "boxes", "labels", "scores"])

# visualize in new notebook