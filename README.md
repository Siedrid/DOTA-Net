# Horizontal Bounding Box Object Detection: DOTA Dataset
The proposed Model for a Horizontal Bounding Box Object Detection of the DOTA Dataset consists of a FasterRCNN architecture with a ResNet50-FPN backbone and the ResNet50 default weights. The model was trained with 12653 image chips. 3818 Image chips were used for validation and testing.

The final model has an mAP of 0.32. In the model pipeline (`02-DOTA_FasterRCNN.py`) Early Stopping is implemented, which terminated the trainig process after 21 epochs. The dataset has a difficulty tag, which was used to remove difficult objects from the dataset. When the difficult objects were kept, the model already terminated after 12 epochs with a mAP of 0.259.
The model is stored in the repository's model folder.

## Tensorboard
The tfevent file for visualization of the training process for the two experiments carried out can be found under `experiments`. When launching TensorBoard set the logdir to `~/DOTA-Net/experiments/`.

## Dataset Preprocessing
The original dataset is accesible via https://datasetninja.com/dota#download.  To perform a HBB Object Detection, the annotations were adjusted by Dr. Hoeser. The respective pipeline is described [here](https://github.com/thho/course_material_04_geo_oma24/blob/main/notebooks/04-hoes_th-DOTA_dataset_prep.ipynb).

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
If you would need any additional libraries, you can install them in the same way.

## For Users (Inference)
The present repository offers the possibility of applying the Object Detection to an unseen dataset. The respective pipeline is exemplarily applied to the test-dev split of the DOTA-Dataset in the `03_Inference.py` script. The pipeline consists of the following steps:
1. Loading the model
2. Preprocessing the images of the test-dev split
3. Preparing the Dataset Class
4. Apply the model to the Dataset and write the new annotations to a csv file.

### Instructions for Inference
The Inference was performed for the test-dev split of the entire DOTA set and its subset. The annotations can be found on Terrabyte: /dss/dsstbyfs02/pn49ci/pn49ci-dss-0022/users/di38tac/DATA/SlidingWindow/dota/test-dev/Inference/FasterRCNN-exp_003_predictions.csv. Inference samples can be visually explored in `03-DOTA_inverence_visualizations.ipynb`.

If you want to perform the inference for your own dataset, change the highlighted paths in the `03_Inference.py` script.

## For Developers
The main training pipeline is accessible via running the `02-DOTA_FasterRCNN.py` script. It considers already preprocessed RGB images with a size of 1024x1024 as well as if the images have differing sizes. In that case the preprocessing pipeline is started before the training pipeline.

The functionalities of the preprocessing module, which can be found in `utils/preprocess_dota.py` is explored in the notebook `01-DOTA_explore_Dataset.ipynb`.

The pipeline consists of the following steps:
1. Setting up the model architecture.
2. Preprocessing Checkpoint.
3. Setup of the Datasets and Dataloaders for Train and Test Splits.
4. Setup of the Summary and Checkpoint Writers.
5. Main Training Pipeline.

### Instructions for training and validation
As the DOTA Dataset is already preprocessed and stored on TB under my USER_PATH, there is no need for you to process it again. Therefore the only thing you have to change in he `02-DOTA_FasterRCNN.py` script is the `writer_path`. If you still want to preprocess it yourself, please change the `USER_PATH` as well.

### Running the Training as SLURM Job on Terrabyte
For Training of >10 epochs it is recommended to start a SLURM Job on Terrabyte. In that case, please make the necessary adjustments int the `02-DOTA_model_training.cmd` file. Please also make sure that the logfile directory is created in advance. Run the scrip from the root of the project directory in MobaXterm by executing:
```
cd DOTA-Net
sbatch code/02-DOTA_model_training.cmd
```

## Discussion
### Model Performance
The model performance is visuallized in `03-DOTA_inference_visualizations.ipynb` for the different classes. The figure shows that the model performs above the mAP of 0.32 for the classes planes, tennis courts and storage tanks. Most classes have an average AP of 0.3 to 0.4, e.g. roundabouts, basketball couts, baseball diamonds, ships and soccer fields.
The AP is 0 for container cranes and helipads.

### Potential failure cases
A potential explanation for the different performance of the model for the different objects is there number during training vs. inference. The most frequent objects are planes, vehicles. This could be accounted for by oversampling underrepresented classes before the training pipeline. The differing Ground Sampling Distances of the images have probably also an impact on the performance. This could be accounted for by adding more transformations to the train_transforms function, like different random zooms.

### Possible Improvements

## References

### Code:
https://github.com/thho/course_material_04_geo_oma24