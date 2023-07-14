# MLiPPaAProject

This repository contains the code developed as part of the final project for the course ''Machine Learning in Particle Physics and Astronomy'' at Radboud Universiteit.
It consists of two parts, one to generate data simulating trajectories for secondary particles of a particle collider and a second one to train a transformer model based on the generated data to reconstruct the trajectory parameters.

## File/Folder Descriptions

### models
Contains all pretrained models of the sweeps and the baseline

### figures
Contains all figures used in this project and the report

### predictions 
Contain predictions files made from the baseline models for 2D and 3D data as well
as from the best models of each sweep. 

### output_*.txt and parameter_*.txt files
Contain the data used for this project. The `*_baseline.txt` are the files
that only contain events with 3 tracks each. The others contain between
0 and 20 tracks per event. `output_*.txt` files contain the points
of the secondary particles hitting the detectors, while `parameter_*.txt` files
contain the angles of the trajectories. The later file therefore can be considered
as the labels file.

### constants.py
This file contains all the constants that are valid throughout the whole repository. Most important constant is the dimensionality constant, that allows changing between two dimensional and three dimensional data

### dataset.py
This file contains an inherited class of the pytorch dataset class, used to load the data for the transformer.

### generate_data.py
This file contains a script to generate the data as well as plot two dimensional data

### train.py
This file contains the script to train the transformer model as well as hyperparameter tune it using WanDB.

### predict.py
This file contains a script that automatically predicts for the baseline model and the best model of the sweep. Depends 
`DIMENSION` in `constants.py`.

### transformer.py
This file contains the simple transformer model for the trajectory parameter reconstruction.

### utils.py
This file contains multiple helper functions for both the data generation as well as data preprocessing and training of the transformer model. 

## How to run

Keep the file structure as is!

Install the libraries as defined in `requirements.txt`. 

### generate_data.py

Set the parameters desired in `constants.py`. Currently, the parameters are set up so 50,000 events 
with between 2 and 20 tracks per event are produced. For a fixed number
of tracks per event set `MIN_NR_TRACKS` equal to `MAX_NR_TRACKS`.
Run

```bash
python3 generate_data.py
```

The data will be stored in the files defined in `DATA_PATH` and `LABEL_PATH`.

### Train a model

#### Model with specific hyperparameters

To train a model with specific hyperparameters, you need to set the following
parameters in `constants.py`:
- `TRAIN` needs to be set to `True`
- `RANDOM_SEARCH` needs to be set to `False`

Optionally the following parameters can be adjusted:
- `BATCH_SIZE`
- `ENCODER_LAYERS`
- `D_MODEL`
- `HEAD`
- `DIM_FEEDFORWARD`
- `DROPOUT`

Afterwards run:
```bash
python3 train.py
```

#### Train multiple models performing random search

**Warning**: For this you need to set your wandb API key first, otherwise this will not work!

After setting up your wandb API key, the following parameters in 
`constants.py` need to be set:
- `TRAIN` needs to be set to `True`
- `RANDOM_SEARCH` needs to be set to `True`

Optionally, the `SWEEP_CONFIGURATION` can be adjusted to desired sweeps.
This training can take multiple hours to days. 

Afterwards run:
```bash
python3 train.py
```

### Evaluate a model on the test set

This requires an already trained model in the `models` directory. 
Models will be automatically saved during training. 

For this the following parameters in `constant.py` need to be set:

- `TRAIN` needs to be set to `False`
- `RANDOM_SEARCH` needs to be set to `False`
- `MODEL_NAME` needs to be set to the name of the trained model in the `models` directory.
- The following parameters need to be set to the correct values the model
was trained with: `BATCH_SIZE`, `ENCODER_LAYERS`, `D_MODEL`, `HEAD`, `DIM_FEEDFORWARD`, `DROPOUT`.
If this is not done, an error will occur. 

Afterwards run:
```bash
python3 train.py
```

### Predict

This requires an already trained model in the `models` directory. 
Models will be automatically saved during training. 

The following parameters in the `constants.py` need to be set to the correct values the baseline model
was trained with: `BATCH_SIZE`, `ENCODER_LAYERS`, `D_MODEL`, `HEAD`, `DIM_FEEDFORWARD`, `DROPOUT`.
If this is not done, an error will occur. 

Afterwards run:
```bash
python3 predict.py
```

This script is set up to run, if the repository is cloned and the `output_2d.txt`
and `parameter_2d.txt` or the respective 3d version is generated/provided. 
It will produce predictions for the best model and the baseline, to
produce it for both 3d and 2d data, change the `DIMENSION` parameter in `constants.py`
and rerun again. 