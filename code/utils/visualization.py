#%%
import pandas as pd
import matplotlib.pyplot as plt
import os 
from pathlib import Path
import matplotlib.patches as patches
import numpy as np
import json
from itertools import cycle
import random
from collections import Counter

DOTA_SET = 'dota' # possible values: dota-subset, dota
SPLIT = 'train' # possible values: train, val, test-dev

DATA_ROOT = Path('/dss/dsstbyfs02/pn49ci/pn49ci-dss-0022/data/')
DOTA_ROOT = DATA_ROOT / DOTA_SET

META_FILE = DOTA_ROOT / 'meta.json'
with open(META_FILE) as f:
    meta = json.load(f)  # Example: { "plane": 1, "ship": 2, ... }
hotkey_to_color = {cls['hotkey']: cls['color'] for cls in meta['classes']}
class_name = {cls['hotkey']: cls['title'] for cls in meta['classes']}

#%%
# plot samples with boxes
def plot_sample(df, ROOT, n=1):
    """
    Plot a sample image with boxes and class legend.

    Parameters:
    df (pandas.Dataframe): Dataframe with the annotations, column names are boxes, scores, labels.
    ROOT (string): Path to the root of the image directory for the respective images.
    """
    rdm_img = df.sample(n=1).iloc[0]
    img_split_path = ROOT / "img" / rdm_img['image_id']

    if pd.isna(rdm_img['boxes']) or pd.isna(rdm_img['labels']):
        boxes_split = []
        classes_split = []
        scores_split = []
    else:
        boxes_split = rdm_img['boxes'].split(';')
        classes_split = list(map(int, rdm_img['labels'].split(';')))
        scores_split = rdm_img['scores'].split(';')

    img = plt.imread(img_split_path)

    fig, ax = plt.subplots()
    ax.imshow(img)

    legend_patches = {}
    #ax.set_title(f"Splitted Image: {rdm_split['tile_filename']}.")
    for box, class_id, score in zip(boxes_split,classes_split, scores_split):
        x_min, y_min, x_max, y_max = map(float, box.split())
        width = x_max - x_min
        height = y_max - y_min
        color = hotkey_to_color[int(class_id)]

        if float(score) < 0.5:
            alpha = 0.5
        else: 
            alpha = 1

        rect = patches.Rectangle(
            (x_min, y_min), width, height, 
            linewidth=2, edgecolor=color, facecolor='none', alpha=alpha)
        ax.add_patch(rect)

        if class_id not in legend_patches:
            legend_patches[class_id] = patches.Patch(
                color=color, 
                label=class_name[int(class_id)])
    if legend_patches:
        ax.legend(handles=list(legend_patches.values()), loc="upper right", title="Classes")

    plt.show()

def plot_class_mAP(df):
    """
    Creates and saves BarChart of the per class AveragePrecision and AverageRecall.

    Parameters:
    df (pandas.DataFrame): Pandas Dataframe with the columns "Class", "AP", "mAR" and "mAP" (mean Average Precision over all Classes).
    """
    df = df.sort_values(by="AP", ascending=False)
    colors = [hotkey_to_color[int(class_id)] for class_id in df.Class]
    class_names = [class_name[int(class_id)] for class_id in df.Class]

    fig, axes = plt.subplots(1,2, figsize=(17,6))
    axes[0].bar(class_names, df.AP, color = colors)
    axes[0].set_title("Per Class Average Precision")
    axes[0].hlines(y=df.mAP[0], xmin= class_names[0], xmax=class_names[-1], linestyle='dashed')    
    axes[0].tick_params("x", rotation=90)

    axes[1].bar(class_names, df.mAR, color = colors)
    axes[1].set_title("Per Class Average Recall")
    axes[1].tick_params("x", rotation=90)

    plt.tight_layout()
    plt.savefig("../media/barchart_mAP_mAR-DOTA.png", format="png")
    plt.show()

def plot_class_freq(splits, dota_set):
    """
    Creates a bar chart of the number of objects per class for a specific split in a Dota Dataset.

    Parameters:
    splits (list): List of strings of the splits to visualize.
    dota_set (string): Name of the Dataset, options: "dota" or "dota-subset".
    """
    ROOT = Path("/dss/dsstbyfs02/pn49ci/pn49ci-dss-0022")
    DOTA_ROOT = ROOT / "users/di38tac" / "DATA"/ "SlidingWindow" / dota_set

    fig, ax = plt.subplots(1, len(splits), figsize=(17,6))
    
    for i, SPLIT in enumerate(splits):
        if SPLIT == 'test-dev':
            csv_file = DOTA_ROOT / SPLIT / "Inference" / f"FasterRCNN-exp_003_predictions.csv"
        else:
            csv_file = DOTA_ROOT / SPLIT / 'ann' / 'annotations.csv'
        df = pd.read_csv(csv_file)
        class_strings = df['labels'].dropna().astype(str)
        all_numbers = [num for row in class_strings for num in row.split(';')]

        freq_counter = Counter(all_numbers)

        freq_df = pd.DataFrame(freq_counter.items(), columns=['class_value', 'frequency'])
        freq_df = freq_df.sort_values(by='frequency').reset_index(drop=True)
        class_names = [class_name[int(class_id)] for class_id in freq_df.class_value]
        colors = [hotkey_to_color[int(class_id)] for class_id in freq_df.class_value]

        ax[i].bar(class_names, freq_df.frequency, color=colors)
        ax[i].set_title(f"Class Balance for {dota_set}/{SPLIT}")
        ax[i].tick_params("x", rotation=90)

def plot_org_and_split(img_name, DOTA_ROOT, DOTA_PREP_ROOT, SPLIT):
    """
    Plots the original image and a chip of the original image after preprocessing.

    Parameters:
    img_name (string): name of the original image, usually "Pxxxx.png".
    DOTA_ROOT (string): Path to the original Root of the Dota Dataset.
    DOTA_PREP_ROOT (string): Path to the Root of the preprocessed Dota Dataset.
    SPLIT (string): Split of the Dota Dataset, options: "test", "val".
    """
    new_annotations = pd.read_csv(DOTA_PREP_ROOT / SPLIT / 'ann/annotations.csv')
    splits = new_annotations[new_annotations['original_image'] == img_name]
    if len(splits) == 0:
        print(f"Image {img_name} not found.")
    else:
        print(f"{len(splits)} Splits of original image {img_name} found.")
    rdm_split = splits.sample(n=1).iloc[0]

    img_split_path = DOTA_PREP_ROOT / SPLIT / "img" / rdm_split['tile_filename']
    img = plt.imread(img_split_path)

    fig, ax = plt.subplots(1,2)
    ax[0].imshow(img)
    ax[0].set_title(f"Splitted Image: {rdm_split['tile_filename']}.")

    if pd.isna(rdm_split['boxes']):
        boxes_split = None
        classes_split = None

    else:
        boxes_split = rdm_split['boxes'].split(';')
        classes_split = rdm_split['labels']
        for box in boxes_split:
            x_min, y_min, x_max, y_max = map(float, box.split())
            width = x_max - x_min
            height = y_max - y_min
            color = "green" #hotkey_to_color[int(class_hotkey)]
            rect = patches.Rectangle((x_min, y_min), width, height, linewidth=2, edgecolor=color, facecolor='none')
            ax[0].add_patch(rect)
    
    # original df
    df = pd.read_csv(DOTA_ROOT / f'{SPLIT}_split.csv')
    row = df[df['img_file'] == img_name].iloc[0]
    img_file = row['img_file']
    bboxes = row['bbox'].split(';')
    classes = row['class'].split(';')

    img_file_path = DOTA_ROOT / SPLIT / 'img' / img_file
    org_img = plt.imread(img_file_path)

    ax[1].imshow(org_img)
    ax[1].set_title(f"Original Image {img_name}")
    for box in bboxes:
        x_min, y_min, x_max, y_max = map(float, box.split())
        width = x_max - x_min
        height = y_max - y_min
        color = "green" #hotkey_to_color[int(class_hotkey)]
        rect = patches.Rectangle((x_min, y_min), width, height, linewidth=2, edgecolor=color, facecolor='none')
        ax[1].add_patch(rect)
    plt.show()