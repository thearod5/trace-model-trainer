import os

from dotenv import load_dotenv

load_dotenv()

OUTPUT_PATH = os.path.expanduser(os.environ["OUTPUT_PATH"])

# Training
TARGET_COL = "t_text"
RANDOM_SEED = 42
BATCH_SIZE = 8
LEARNING_RATE = 5e-6
TEST_SIZE = 0.25
N_EPOCHS = 1
DEFAULT_FP16 = True

# MODELS
DEFAULT_MLM_MODEL = "bert-base-uncased"
DEFAULT_ST_MODEL = "all-MiniLM-L6-v2"
