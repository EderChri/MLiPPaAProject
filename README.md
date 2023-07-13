# MLiPPaAProject

This repository contains the code developed as part of the final project for the course ''Machine Learning in Particle Physics and Astronomy'' at Radboud Universiteit.
It consists of two parts, one to generate data simulating trajectories for secondary particles of a particle collider and a second one to train a transformer model based on the generated data to reconstruct the trajectory parameters.

## constants.py
This file contains all the constants that are valid throughout the whole repository. Most important constant is the dimensionality constant, that allows changing between two dimensional and three dimensional data

## dataset.py
This file contains an inherited class of the pytorch dataset class, used to load the data for the transformer.

## generate_data.py
This file constains a script to generate the data as well as plot two dimensional data

## train.py
This file contains the script to train the transformer model as well as hyperparameter tune it using WanDB.

## transformer.py
This file contains the simple transformer model for the trajectory parameter reconstruction.

## utils.py
This file contains multiple helper functions for both the data generation as well as data preprocessing and training of the transformer model. 
