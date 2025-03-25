import pandas as pd
import PIL.Image
import torchvision.transforms.functional as TF
from pathlib import Path
import os

def preprocess_dota_dataset(csv_file, root_img_dir, output_dir, tile_size=1024, overlap=200, boxes=None):
    # what happens with tiles without boxes?
    # what happens with boxes that are split by the window?
    # option: inference = True, then dont care about the boxes

    img_dir = output_dir / "img"
    ann_dir = output_dir / "ann"

    os.makedirs(img_dir, exist_ok=True)
    os.makedirs(ann_dir, exist_ok=True)

    output_csv = ann_dir / "annotations.csv"
    new_annotations = []

    annotations = pd.read_csv(csv_file).reset_index(drop=True)

    for idx in range(len(annotations)):
        img_name = annotations.iloc[idx, 0]

        img_path = Path(root_img_dir) / img_name
        img = PIL.Image.open(img_path).convert("RGB")
        
        img, pad_left, pad_top = pad_image(img, tile_size)

        if boxes is not None:
            boxes_string = str(annotations.iloc[idx, 1])
            labels_string = str(annotations.iloc[idx, 2])            
            boxes = [list(map(int, box.split())) for box in boxes_string.split(";") if box != 'nan']
            labels = [int(label) for label in labels_string.split(';') if label.strip().isdigit()]

            # Adjust bounding boxes for padding
            boxes = [[x_min + pad_left, y_min + pad_top, x_max + pad_left, y_max + pad_top] for x_min, y_min, x_max, y_max in boxes]
        
        tiles = split_image(img, tile_size, overlap, boxes=boxes)
        
        for i, tile in enumerate(tiles):
            tile_img = tile[0]
            tile_basename = os.path.splitext(os.path.basename(img_name))[0]
            tile_filename = f"{tile_basename}_{i}.png"
            tile_path = img_dir / tile_filename
            tile_img.save(tile_path)

            if boxes is not None:
                tile_img, tile_boxes, box_indices = tile
                tile_labels = [labels[i] for i in box_indices]

                new_box_string = ";".join([" ".join(map(str, box)) for box in tile_boxes])
                new_label_string = ";".join(map(str, tile_labels))

                annotations = [tile_filename, img_name, new_box_string, new_label_string]
            else:
                annotations = [tile_filename, img_name, "nan", "nan"]
            new_annotations.append(annotations)

    # Save new annotations
    df_new = pd.DataFrame(new_annotations, columns=["tile_filename", "original_image", "boxes", "labels"])
    df_new.to_csv(output_csv, index=False)
    print(f"Processed dataset saved at: {output_csv}")


# pad before this function
def split_image(img, tile_size, overlap, boxes = None):
    # img is already padded if smaller than tile size

    w, h = img.size  # Updated size after padding
    stride = tile_size - overlap
    tiles = []

    for y in range(0, h-overlap, stride):
        for x in range(0, w-overlap, stride):
            tile_img = TF.crop(img, y, x, tile_size, tile_size)
            #tiles.append((tile_img, x, y))
            if boxes is not None:
                tile_boxes, box_indices = adjust_boxes(boxes, x, y, tile_size)
                yield tile_img, tile_boxes, box_indices, x, y
                tiles.append((tile_img, tile_boxes, box_indices))

            else:
                tiles.append((tile_img)) # also x and y
            return tiles
        
def adjust_boxes(boxes, tile_x, tile_y, tile_size):# add iou threshold
    tile_boxes, box_indices = [], []
    for i, (x_min, y_min, x_max, y_max) in enumerate(boxes):
        x_min, y_min, x_max, y_max = x_min - tile_x, y_min - tile_y, x_max - tile_x, y_max - tile_y
        x_min, y_min, x_max, y_max = max(0, x_min), max(0, y_min), min(tile_size, x_max), min(tile_size, y_max)
        if x_max > x_min and y_max > y_min:
            tile_boxes.append([x_min, y_min, x_max, y_max])
            box_indices.append(i)
    return tile_boxes, box_indices       

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

