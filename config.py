from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent

# ---- Dataset paths (Kaggle dataset: sabahesaraki/2017-2017) ----
DATA_ROOT = Path("/kaggle/input/datasets/sabahesaraki/2017-2017")
ANN_DIR = DATA_ROOT / "annotations_trainval2017" / "annotations"

TRAIN_IMAGES_DIR = DATA_ROOT / "train2017" / "train2017"
TRAIN_CAPTIONS_JSON = ANN_DIR / "captions_train2017.json"
TRAIN_INSTANCES_JSON = ANN_DIR / "instances_train2017.json"

# Test set comes from COCO val2017 (completely separate from train2017).
TEST_IMAGES_DIR = DATA_ROOT / "val2017" / "val2017"
TEST_CAPTIONS_JSON = ANN_DIR / "captions_val2017.json"

# ---- Outputs ----
_KAGGLE_WORKING = Path("/kaggle/working")
OUTPUT_DIR = (_KAGGLE_WORKING if _KAGGLE_WORKING.exists() else PROJECT_ROOT) / "outputs"
PROCESSED_DIR = OUTPUT_DIR / "processed"
REPORTS_DIR = OUTPUT_DIR / "reports"
FIGURES_DIR = OUTPUT_DIR / "visualizations"
EMBEDDINGS_DIR = OUTPUT_DIR / "embeddings"
INDEX_DIR = OUTPUT_DIR / "faiss"

RANDOM_SEED = 42

# ---- Split policy ----
KB_COUNT = 10000          # knowledge base size (clean, unique, random sample of train2017)
CANDIDATE_POOL = 10500    # random images we audit; the first 10,000 valid ones are kept
VALIDATION_COUNT = 100    # drawn from WITHIN the 10,000 knowledge-base images
TEST_COUNT = 100          # from val2017, separate from the knowledge base

# ---- Embeddings ----
CLIP_MODEL_NAME = "openai/clip-vit-base-patch32"
BATCH_SIZE = 64
NUM_WORKERS = 2
DEVICE = "cuda"           # falls back to CPU automatically
NEAR_DUPLICATE_THRESHOLD = 0.95   # CLIP cosine similarity
EMBEDDING_IMAGE_LIMIT = None      # set e.g. 200 for a quick smoke test
