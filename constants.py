from torch import nn

DIMENSION = 3  # dimensionality of data
RANDOM_SEARCH = False  # Trains single model or evaluates existing model on test set when False depending on TRAIN
TRAIN = True  # Loads model and evaluates on test set when False
DATA_PATH = f"output_{DIMENSION}d.txt"
LABEL_PATH = f"parameter_{DIMENSION}d.txt"
TEST_BATCH_SIZE = 2
EPOCHS = 100
PADDING_LEN_INPUT = 100
PADDING_LEN_LBL = 20
PAD_TOKEN = 50
EARLY_STOPPING = 5
LOSS_FN = nn.MSELoss()
# Only relevant for data generation
NR_EVENTS = 50000  # number of events to produce
DETECTOR_RADII = [1, 2, 3, 4, 5]
NR_DETECTORS = len(DETECTOR_RADII)
MAX_NR_TRACKS = 20  # maximal number of tracks per event
MIN_NR_TRACKS = 2  # minimal number of tracks per event
# Only relevant for loading models or training a specific one
## For loading these parameters need to be set to the same values as they were during training!
MODEL_NAME = "transformer_encoder_best_3d_baseline"
BATCH_SIZE = 256
ENCODER_LAYERS = 1
D_MODEL = 32
HEAD = 1
DIM_FEEDFORWARD = 1
DROPOUT = 0.1
# Only relevant if random search is true
# Sweep configuration for wandb
SWEEP_CONFIGURATION = {
    'method': 'random',
    'name': 'sweep',
    'metric': {
        'goal': 'minimize',
        'name': 'val_loss'
    },
    'parameters': {
        'batch_size': {'values': [32, 64, 128, 256]},
        'lr': {'max': 0.001, 'min': 0.00001},
        'dropout': {'values': [0.05, 0.1, 0.15, 0.2, 0.25, 0.3]},
        'd_model': {'values': [16, 32, 64, 128]},
        'head': {'values': [1, 2, 4, 8]},
        'num_encoder_layers': {'values': [2, 4, 8]},
        'dim_feedforward': {'values': [1, 2]},
    }
}
