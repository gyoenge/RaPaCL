from __future__ import annotations

CATEGORICAL_COLUMNS: list[str] = None 
NUMERICAL_COLUMNS: list[str] = None
BINARY_COLUMNS: list[str] = None
NUM_CLASSES: int = None

# Column Partitioning 
NUM_PARTITIONS: int = 3 
OVERLAP_RATIO: float = 0.5

# TransTab Encoder Architecture  
HIDDEN_DIM: int = 128
NUM_LAYERS: int = 2
NUM_ATTENTION_HEADS: int = 8
FFN_DIM: int = 256
ACTIVATION: str = "relu"
DROPOUT_PROB: float = 0.0

# Classifier Architecture
CLASSIFIER_FFN_DIM: int = 128

# Training Hyperparameters
NUM_EPOCHS: int = 50 
BATCH_SIZE: int = 64
LEARNING_RATE: float = 1e-4
WEIGHT_DECAY: float = 0.0
PATIENCE: int = 5
WARMUP_RATIO: float = None
WARMUP_STEPS: int = None

# Evaluation Settings
EVAL_BATCH_SIZE: int = 256
EVAL_METRICS: str = "auc"
