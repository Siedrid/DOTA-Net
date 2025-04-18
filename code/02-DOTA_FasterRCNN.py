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

import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor, FasterRCNN_ResNet50_FPN_Weights
from torchinfo import summary
from torch.optim import SGD
from torch.utils.tensorboard import SummaryWriter
from torchmetrics.detection.mean_ap import MeanAveragePrecision

import sys
sys.path.append('/dss/dsshome1/0A/di38tac/DOTA-Net/code')
from utils.dataset import DOTA_DATASET_v2, DOTA_preprocessed
from utils.training import prepare_map_predictions, prepare_map_targets, EarlyStopping
from utils.preprocess_dota import DotaPreprocessor

DOTA_SET = 'dota' # possible values: dota-subset, dota
dota_splits = ['train', 'val', 'test-dev']

ROOT = Path("/dss/dsstbyfs02/pn49ci/pn49ci-dss-0022")
DATA_ROOT = ROOT / 'data'
DOTA_ROOT = DATA_ROOT / DOTA_SET

######################################################################################
# if you want to preprocess the data yourself change this to your username
# other wise keep mine, as I already preprocessed the data and stored it in my USER_PATH
USER = "di38tac"
USER_PATH = ROOT / f"users/{USER}"

# path to write the tfeventfiles and checkpoints to, change this
writer_path = Path("/dss/dsshome1/0A/di38tac/DOTA-Net/FasterRCNN")
######################################################################################

# Global Variables
model_name = 'FasterRCNN'
num_epochs = 100
TRAIN_BATCH_SIZE = 64
VAL_BATCH_SIZE = 64
VAL_SCORE_THRESHOLD = 0.5
EXPERIMENT_ID = "exp_003"
PREPROCESSING = False # True if you want to preprocess the data to 1024x1024 image chips first

# Model Setup -------------------------------------------
model = torchvision.models.detection.fasterrcnn_resnet50_fpn(
    trainable_backbone_layers=5,
    weights=FasterRCNN_ResNet50_FPN_Weights.DEFAULT)
num_classes = 19
in_features = model.roi_heads.box_predictor.cls_score.in_features
model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

early_stopping = EarlyStopping(patience=10, delta=0.005)
optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=0.0005)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=1e-6)

# Checkpoint for preprocessing------------------

if PREPROCESSING:
    print("Preprocessing Pipeline started.")

    for SPLIT in dota_splits:

        print(f"Preprocessing {SPLIT} split.")
        out_dir = USER_PATH / "DATA"/ "SlidingWindow" / DOTA_SET / SPLIT
        os.makedirs(out_dir, exist_ok=True)
        csv_file = DOTA_ROOT / f'{SPLIT}_split.csv'
        IMGS_DIR = DOTA_ROOT / SPLIT / 'img'

        preprocessor = DotaPreprocessor(
            csv_file=csv_file,
            root_img_dir=IMGS_DIR,
            output_dir=out_dir,
            tile_size=1024,
            overlap=200,
            boxes=True,
            num_workers=8
        )

        preprocessor.process_all_images()

    DOTA_ROOT = USER_PATH / "DATA"/ "SlidingWindow" / DOTA_SET

else:
    DOTA_ROOT = USER_PATH / "DATA"/ "SlidingWindow" / DOTA_SET

# Transformations
def train_transforms() -> v2.Compose:
        """
        Transformations to be applied to the train dataset samples.

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
                v2.SanitizeBoundingBoxes(),
                v2.ToDtype(torch.float32, scale=True)
            ]
        )

# Datasets and Dataloaders ----------------------------------
from utils.dataset import val_transforms, collate_fn

if PREPROCESSING:
    print("Preparing Dataloaders...")
    train_dataset = DOTA_DATASET_v2(
        csv_file=DOTA_ROOT / 'train_split.csv', 
        root_img_dir=DOTA_ROOT / 'train' / 'img', 
        tile_size=1024, 
        overlap=100, 
        transform=train_transforms(),
        difficult=False
    )

    val_dataset = DOTA_DATASET_v2(
        csv_file=DOTA_ROOT / 'val_split.csv', 
        root_img_dir=DOTA_ROOT / 'val' / 'img', 
        tile_size=1024, 
        overlap=100, 
        transform=val_transforms(),
        difficult=False
    )
    print("Dataloader ready.")

else:
    # Preprocessed DOTA Set Dataloaders
    train_dataset = DOTA_preprocessed(
        csv_file=DOTA_ROOT / 'train' / "ann/annotations.csv",
        root_img_dir=DOTA_ROOT / 'train' / "img",
        transform=train_transforms(),
        difficult=False
    )

    val_dataset = DOTA_preprocessed(
        csv_file=DOTA_ROOT / 'val' / "ann/annotations.csv",
        root_img_dir=DOTA_ROOT / 'val' / "img",
        transform=val_transforms(),
        difficult=False
    )

train_loader = DataLoader(
    train_dataset,
    batch_size=TRAIN_BATCH_SIZE,
    shuffle=True,
    num_workers=4,
    prefetch_factor=2,
    drop_last=True,
    collate_fn=collate_fn,
)

val_loader = DataLoader(
    val_dataset,
    batch_size=VAL_BATCH_SIZE,
    shuffle=True,
    num_workers=4,
    prefetch_factor=2,
    drop_last=True,
    collate_fn=collate_fn,
)

print("Size of Training Data is:", len(train_dataset))
print("Size of Validation Data is:", len(val_dataset))

# Training Pipeline -------------------------------
print("Starting Training Pipeline.")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
_ = model.to(device)
print(f"Device: {device}")


def train(dataloader, model, optimizer, epoch, writer):
    model.train()
    running_loss = 0.0

    for i, (images, targets) in enumerate(dataloader):
        images = [image.to(device) for image in images]
        targets = [{k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in t.items()} for t in targets]

        loss_dict = model(images, targets)
        losses = sum(loss for loss in loss_dict.values()) # computing 4 losses in this architecture
        running_loss += losses.item() # sum up the losses

        optimizer.zero_grad()
        losses.backward()
        optimizer.step()


        if i % 10 == 0:
            print(f"Step [{i}/{len(dataloader)}], Loss: {losses.item():.4f}")

    avg_train_loss = running_loss / len(dataloader)
    
    writer.add_scalar('Loss/train', avg_train_loss, epoch)
    return avg_train_loss

def validate(dataloader, model, epoch, writer):
    model.train()
    running_loss = 0.0

    mAP = MeanAveragePrecision(iou_type="bbox").to(device)

    with torch.no_grad():
        for images, targets in dataloader:
            images = [image.to(device) for image in images]
            targets = [
                {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in t.items()}
                for t in targets
            ]

            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())
            running_loss += losses.item()

            model.eval()
            preds = model(images)
            model.train()

            mAP_preds = prepare_map_predictions(preds, threshold=VAL_SCORE_THRESHOLD)
            mAP_targets = prepare_map_targets(targets)
            mAP.update(mAP_preds, mAP_targets)

    avg_val_loss = running_loss / len(dataloader)
    mAP_values = mAP.compute()

    writer.add_scalar('Loss/val', avg_val_loss, epoch)
    writer.add_scalar('mAP/val', mAP_values['map'], epoch)
    writer.add_scalar('mAP_50/val', mAP_values['map_50'], epoch)
    writer.add_scalar('mAP_75/val', mAP_values['map_75'], epoch)
    writer.add_scalar('mAR_10/val', mAP_values['mar_10'], epoch)

    print(f'Validation Loss: {avg_val_loss}')
    print(f'mAP: {mAP_values["map"]}')
    print(f'mAP_50: {mAP_values["map_50"]}')
    print(f'mAP_75: {mAP_values["map_75"]}')
    return avg_val_loss

# Main function -----------------------------

# setup writer
EXPERIMENT_GROUP = f"{DOTA_SET}_{model_name}" # subfolder for model
EXPERIMENT_DIR = writer_path / f"experiments/{EXPERIMENT_GROUP}"
EXPERIMENT_DIR.mkdir(parents=True, exist_ok=True)

CHECKPOINTS_DIR = writer_path / f"checkpoints/{EXPERIMENT_GROUP}"
CHECKPOINTS_DIR.mkdir(parents=True, exist_ok=True)

writer = SummaryWriter(EXPERIMENT_DIR / EXPERIMENT_ID)

best_val_loss = float('inf')
best_checkpoint_path = None

# the val score threshold is used to determine if a prediction should be considered

for epoch in range(num_epochs):
    avg_train_loss = train(train_loader, model, optimizer, epoch, writer)
    print(f'Epoch [{epoch}/{num_epochs-1}], Loss: {avg_train_loss}')

    avg_val_loss = validate(val_loader, model, epoch, writer)
    scheduler.step()

    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        best_checkpoint_path = CHECKPOINTS_DIR / f'{EXPERIMENT_GROUP}_{EXPERIMENT_ID}_epoch_{epoch}.pth'
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'loss': best_val_loss,
        }, best_checkpoint_path)
        print(f'Checkpoint saved at {best_checkpoint_path}')
        best_checkpoint_path_str = "val_loss:" + str(best_val_loss) + "@" + str(best_checkpoint_path)
        writer.add_text("Best Checkpoint Path", str(best_checkpoint_path_str), epoch)

    # Early Stopping
    early_stopping(avg_val_loss, model)
    if early_stopping.early_stop:
        print("Early Stopping")
        break

writer.close()

#%%