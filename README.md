# Horizontal Bounding Box Object Detection: DOTA Dataset
The original dataset is accesible via ... To perform a HBB Object Detection, the annotations were adjusted by Dr. Hoeser. The respective pipeline is described here:

## Environment Setup
Install uv and MobaXTerm and open MobaXTerm
Login to Terrabyte
cd to project directory
uv venv --python 3.11
uv pip install numpy pandas torch torchvision torchmetrics tensorboard scikit-learn ipykernel
uv pip install torchmetrics[detection] # for MAP to work

https://docs.astral.sh/uv/pip/environments/#creating-a-virtual-environment


## For Users (Inference)
Think about what to do with the test-dev split.
output is a new csv with the bounding boxes and labels?

## For Developers
The main training pipeline is accessible via running the 02-DOTA_FasterRCNN.py script. It considers already preprocessed RGB images with a size of 1024x1024 as well as if the images have differing sizes. In that case the preprocessing pipeline is started before the training pipeline.

The functionalities of the preprocessing pipeline can be explored in the notebook 01-DOTA_explore_Dataset.ipynb.

### Running the Training as SLURM Job on Terrabyte
For Training of >10 epochs it is recommended to start a SLURM Job on Terrabyte. In that case, please make the necessary adjustments int the 02-DOTA_model_training.cmd file an run it in e.g. MobaXterm from the root of the project directory.


## References

### Code:
https://github.com/thho/course_material_04_geo_oma24