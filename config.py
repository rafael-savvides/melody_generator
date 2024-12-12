DEVICE = "mps"
SEED_SPLIT = 42
SEED_LOADER = 42

# Data
STEP_SIZE = 0.25
SEQUENCE_LENGTH = 64
TOKENS = {"rest": "R", "hold": "H", "end": "E"}

# Training
PCT_TR = 0.8
LEARNING_RATE = 0.001
BATCH_SIZE = 16
NUM_EPOCHS = 100
DATA_SIZE = None  # If None, use all data.
OPTIMIZER = "Adam"
OPTIMIZER_PARAMS = {"weight_decay": 1e-5}

# Model
EMBEDDING_SIZE = 32
HIDDEN_SIZE = 256
OUTPUT_SIZE = None  # If None, use all encodable strings, i.e., len(encoding).
DROPOUT = 0.2  # At LSTM output

# Paths
PATH_TO_DATA = "data"
PATH_TO_MODELS = "models"
PATH_TO_LOGS = "logs"

DATASETS = {
    "maestro-v3.0.0-time_series": {
        # maestro-v3.0.0 has 3696777 notes in 1276 files
        "raw": "maestro-v3.0.0",
        "processed": "maestro-v3.0.0-time_series",
        "encoding_path": "encoding.json",
        "notes": [],  # TODO
    },
    "jsb_chorales": {
        "raw": "jsb-chorales-16th.json",
        "processed": "jsb_chorales.txt",
        "encoding_path": "encoding_jsb.json",
        "notes": [str(i) for i in range(36, 82)] + [TOKENS["rest"], TOKENS["end"]],
    },
}
