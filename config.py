DEVICE = "mps"
SEED_SPLIT = 42
SEED_LOADER = 42

# Data
NUM_PITCHES = 128
STEP_SIZE = 0.25
TOKENS = {"rest": "R", "hold": "H", "end": "E"}
SEQUENCE_LENGTH = 64
PCT_TR = 0.8

# Training
LEARNING_RATE = 0.08
BATCH_SIZE = 16
NUM_EPOCHS = 50
NUM_FILES = 50

# Model
NUM_UNIQUE_TOKENS = NUM_PITCHES + len(TOKENS)  # TODO Rename to output_size?
EMBEDDING_SIZE = 32
HIDDEN_SIZE = 256

# Paths
PATH_TO_DATA = "data"
PATH_TO_MODELS = "models"
PATH_TO_ENCODING = "encoding.json"

DATASETS = {
    "maestro-v3.0.0-time_series": {
        # maestro-v3.0.0 has 3696777 notes in 1276 files
        "raw": "maestro-v3.0.0",
        "processed": "maestro-v3.0.0-time_series",
    },
    "jsb_chorales": {
        "raw": "jsb-chorales-16th.json",
        "processed": "jsb_chorales.txt",
    },
}
