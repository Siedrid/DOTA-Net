#%%
from pathlib import Path
from typing import Callable, Optional, Tuple, Dict, List
import pandas as pd
import numpy as np 
import PIL
from PIL import Image
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

# Path to the original data, change this to preprocess your own data
###########################################################
ROOT = Path("/dss/dsstbyfs02/pn49ci/pn49ci-dss-0022")
DATA_ROOT = ROOT / 'data'
DOTA_ROOT = DATA_ROOT / DOTA_SET
IMGS_DIR = DOTA_ROOT / SPLIT / 'img'
csv_file = DOTA_ROOT / f'{SPLIT}_split.csv'
###########################################################

print("Number of Images to preprocess: ", len(os.listdir(IMGS_DIR)))

# in this directory the preprocessed images and annotations are stored
# if you want to process your own data, change this to the desired output directory for your preprocessed data
# and set preprocess to True
USER = "di38tac"
USER_PATH = ROOT / f"users/{USER}"
out_dir = USER_PATH / "DATA"/ "SlidingWindow" / DOTA_SET / SPLIT
os.makedirs(out_dir, exist_ok=True)
preprocess=False

# change this to match your root project directory to load the model
##########################################################
home = Path("/dss/dsshome1/0A/di38tac/DOTA-Net") 
##########################################################
model_name = 'FasterRCNN'
EXPERIMENT_GROUP = f"dota_{model_name}" 
EXPERIMENT_ID = "exp_003"

EXPERIMENT_DIR = home / model_name / f"experiments/{EXPERIMENT_GROUP}" / EXPERIMENT_ID
exp_fls = os.listdir(EXPERIMENT_DIR)
checkpoint = EXPERIMENT_DIR / exp_fls[0]

best_checkpoint_path = get_best_checkpoint_path(str(checkpoint))

model = torchvision.models.detection.fasterrcnn_resnet50_fpn()
num_classes = 19
in_features = model.roi_heads.box_predictor.cls_score.in_features
model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

checkpoint = torch.load(best_checkpoint_path, weights_only=False)
model.load_state_dict(checkpoint['model_state_dict'])
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

# Pipeline:
# Preprocess the images 
if preprocess:
    print("Preprocessing dataset...")
    from utils.preprocess_dota import DotaPreprocessor, preprocess_dota_dataset_v0

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

# Create the dataset class
class DOTA_Inference(Dataset):
    def __init__(self, csv_file, root_img_dir, transform=None):
        self.csv_file = csv_file
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
    


# make predictions preprocessed
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
    csv_file=out_dir / "ann" / "annotations.csv",
    root_img_dir=out_dir / "img",
    transform=inference_transforms()
)
print("Size of Inference Data is:", len(inference_dataset))

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
    score_string = ";".join(map(str, scores))

    annotations = [prediction["image_id"], new_box_string, new_label_string, score_string]
    
    predicted.append(annotations)

prediction_df = pd.DataFrame(predicted, columns=["image_id", "boxes", "labels", "scores"])

INFERENCE_DIR = out_dir / "Inference" 
os.makedirs(INFERENCE_DIR, exist_ok=True)

out_path = INFERENCE_DIR / f"{model_name}-{EXPERIMENT_ID}_predictions.csv"
prediction_df.to_csv(out_path, index=False)

print(f"Predictions for {DOTA_SET}/{SPLIT} Dataset are saved to {out_path}.")
#%%
# # visualize in new notebook