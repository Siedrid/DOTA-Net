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
sbatch script


## References

## Code:
https://github.com/thho/course_material_04_geo_oma24