import pandas as pd
import PIL.Image
import torchvision.transforms.functional as TF
from pathlib import Path
import os

def preprocess_dota_dataset(csv_file, root_img_dir, output_dir, tile_size=1024, overlap=200):
    # what happens with tiles without boxes?
    # what happens with boxes that are split by the window?
    
    os.makedirs(output_dir, exist_ok=True)
    output_csv = os.path.join(output_dir, "processed_annotations.csv")
    new_annotations = []

    annotations = pd.read_csv(csv_file).reset_index(drop=True)

    stride = tile_size - overlap

    for idx in range(len(annotations)):
        img_name = annotations.iloc[idx, 0]
        boxes_string = str(annotations.iloc[idx, 1])
        labels_string = str(annotations.iloc[idx, 2])

        img_path = Path(root_img_dir) / img_name
        img = PIL.Image.open(img_path).convert("RGB")
        
        img, pad_left, pad_top = pad_image(img, tile_size)
        w, h = img.size  # Updated size after padding

        boxes = [list(map(int, box.split())) for box in boxes_string.split(";") if box != 'nan']
        labels = [int(label) for label in labels_string.split(';') if label.strip().isdigit()]

        # Adjust bounding boxes for padding
        boxes = [[x_min + pad_left, y_min + pad_top, x_max + pad_left, y_max + pad_top] for x_min, y_min, x_max, y_max in boxes]
        
        for y in range(0, h - overlap, stride):
            for x in range(0, w - overlap, stride):
                tile_boxes, box_indices = adjust_boxes(boxes, x, y, tile_size)
                tile_labels = [labels[i] for i in box_indices]

                if tile_boxes:  # Only save tiles with at least one annotation
                    tile_img = TF.crop(img, y, x, tile_size, tile_size)

                    tile_basename = os.path.splitext(os.path.basename(img_name))[0]
                    tile_filename = f"{tile_basename}_tile_{x}_{y}.png"
                    
                    tile_path = os.path.join(output_dir, tile_filename)
                    tile_img.save(tile_path)

                    new_box_string = ";".join([" ".join(map(str, box)) for box in tile_boxes])
                    new_label_string = ";".join(map(str, tile_labels))
                    
                    new_annotations.append([tile_filename, img_name, new_label_string, new_label_string])
    
    # Save new annotations
    df_new = pd.DataFrame(new_annotations, columns=["tile_filename", "original_image", "boxes", "labels"])
    df_new.to_csv(output_csv, index=False)
    print(f"Processed dataset saved at: {output_csv}")

def pad_image(img, tile_size):
    w, h = img.size
    if w >= tile_size and h >= tile_size:
        return img, 0, 0
    pad_left = max(0, (tile_size - w) // 2)
    pad_top = max(0, (tile_size - h) // 2)
    pad_right = max(0, tile_size - (w + pad_left))
    pad_bottom = max(0, tile_size - (h + pad_top))
    img = TF.pad(img, (pad_left, pad_top, pad_right, pad_bottom), fill=0)
    return img, pad_left, pad_top

def adjust_boxes(boxes, tile_x, tile_y, tile_size):
    tile_boxes, box_indices = [], []
    for i, (x_min, y_min, x_max, y_max) in enumerate(boxes):
        x_min, y_min, x_max, y_max = x_min - tile_x, y_min - tile_y, x_max - tile_x, y_max - tile_y
        x_min, y_min, x_max, y_max = max(0, x_min), max(0, y_min), min(tile_size, x_max), min(tile_size, y_max)
        if x_max > x_min and y_max > y_min:
            tile_boxes.append([x_min, y_min, x_max, y_max])
            box_indices.append(i)
    return tile_boxes, box_indices
