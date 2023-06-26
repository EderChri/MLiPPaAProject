from torch import nn

DETECTOR_RADII = [1, 2, 3, 4, 5]
NR_DETECTORS = len(DETECTOR_RADII)
DIMENSION = 2 # dimensionality of data
DATA_PATH = "output.txt"
LABEL_PATH = "parameter.txt"
BATCH_SIZE = 32
TEST_BATCH_SIZE = 2
EPOCHS = 50
TRAIN = True
PADDING_LEN_INPUT = 100
PADDING_LEN_LBL = 20
PAD_TOKEN = 50
NR_EVENTS = 500
EARLY_STOPPING = 5
LOSS_FN = nn.MSELoss()
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