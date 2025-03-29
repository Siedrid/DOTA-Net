# Horizontal Bounding Box Object Detection: DOTA Dataset
The proposed Model for a Horizontal Bounding Box Object Detection of the DOTA Dataset consists of a FasterRCNN architecture with a ... backbone and ... weights. The model was trained for 100 epochs with 12653 image chips. 3818 Image chips were used for validation and testing.

The final model has an accuracy of ...
The model can be downloaded from this repository.

## Tensorboard

## Dataset Preprocessing
The original dataset is accesible via ... To perform a HBB Object Detection, the annotations were adjusted by Dr. Hoeser. The respective pipeline is described here:

## Environment Setup
For this project the environment by Thorsten Hoeser was used. The only additional library needed is the cv2 library which can be installed into the virtual environment by logging into terrabyte in MobaXterm and activating the venv:
```
cd 04-geo-oma24/course_material_04_geo_oma24
source .venv/bin/activate
```
Then the venv is activated. The cv2 library can be installed by:
```
module load uv
uv pip install opencv-python
```

## For Users (Inference)
Think about what to do with the test-dev split.
output is a new csv with the bounding boxes and labels?

## For Developers
The main training pipeline is accessible via running the `02-DOTA_FasterRCNN.py` script. It considers already preprocessed RGB images with a size of 1024x1024 as well as if the images have differing sizes. In that case the preprocessing pipeline is started before the training pipeline.

The functionalities of the preprocessing pipeline can be explored in the notebook `01-DOTA_explore_Dataset.ipynb`.

### Running the Training as SLURM Job on Terrabyte
For Training of >10 epochs it is recommended to start a SLURM Job on Terrabyte. In that case, please make the necessary adjustments int the `02-DOTA_model_training.cmd` file. Please also make sure that the logfile directory is created in advance. Run the scrip from the root of the project directory in MobaXterm:
```
cd DOTA-Net
sbatch code/02-DOTA_model_training.cmd
```

## References

### Code:
https://github.com/thho/course_material_04_geo_oma24