from typing import Any, Dict, Tuple, List, Optional
from torch.utils.data import DataLoader
from torchvision import tv_tensors
import torch
import pandas as pd
from torchvision.utils import draw_bounding_boxes
import torchvision.transforms.functional as F
import matplotlib.pyplot as plt

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