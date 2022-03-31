# Train Ship Finder
Take the [shipsnet dataset](https://www.kaggle.com/datasets/rhammell/ships-in-satellite-imagery) and run two tasks:
1. **Data generation**: Use the Keras [ImageDataGenerator](https://www.tensorflow.org/api_docs/python/tf/keras/preprocessing/image/ImageDataGenerator) to generate more training data
2. **Model training**: Train a deep convolutional neural network to find ships in images using TensorFlow on GPUs.

Returns the augmented data and a model directory with validation and exploration data.


### Resources:
Modify the `executors.json` file in the workflow's directory to select the resources for tasks (1) and (2) above.

### Requirements:
- Singularity container build from the `singularity.file` in the workflow directory with the command `sudo singularity build tensorflow_latest-gpu-jupyter-extra.sif singularity.file`
- The following conda environment in the local and remote environments: `conda create --name parsl_py39 python=3.9; yes | pip install`. The workflow tries to install it if it is not found.
<br>
<div style="text-align:left;"><img src="https://www.dropbox.com/s/ovh4dhkb0zto3gz/diu_training.png?raw=1" height="200"></div>
<br>
<div style="text-align:left;"><img src="https://www.dropbox.com/s/s6mgr79cuw0rw0d/diu_workflow_diagram.png?raw=1" height="400"></div>