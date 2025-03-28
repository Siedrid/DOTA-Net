import pandas as pd
import matplotlib.pyplot as plt
import os 
from pathlib import Path
import matplotlib.patches as patches
import numpy as np
import json
from itertools import cycle
import random

DOTA_SET = 'dota' # possible values: dota-subset, dota
SPLIT = 'train' # possible values: train, val, test-dev

DATA_ROOT = Path('/dss/dsstbyfs02/pn49ci/pn49ci-dss-0022/data/')
DOTA_ROOT = DATA_ROOT / DOTA_SET

META_FILE = DOTA_ROOT / 'meta.json'
with open(META_FILE) as f:
    meta = json.load(f)  # Example: { "plane": 1, "ship": 2, ... }
hotkey_to_color = {cls['hotkey']: cls['color'] for cls in meta['classes']}

# plot samples with boxes
def plot_sample(df, ROOT, n=1):
    rdm_img = df.sample(n=n).iloc[0]
    img_split_path = ROOT / "img" / rdm_img['image_id']

    if pd.isna(rdm_img['boxes']) or pd.isna(rdm_img['labels']):
        boxes_split = []
        classes_split = []
    else:
        boxes_split = rdm_img['boxes'].split(';')
        classes_split = list(map(int, rdm_img['labels'].split(';')))

    img = plt.imread(img_split_path)

    fig, ax = plt.subplots()
    ax.imshow(img)
    #ax.set_title(f"Splitted Image: {rdm_split['tile_filename']}.")
    for box, class_id in zip(boxes_split,classes_split):
        x_min, y_min, x_max, y_max = map(float, box.split())
        width = x_max - x_min
        height = y_max - y_min
        color = hotkey_to_color[int(class_id)]

        rect = patches.Rectangle((x_min, y_min), width, height, linewidth=2, edgecolor=color, facecolor='none')
        ax.add_patch(rect)

    plt.show()