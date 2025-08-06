from pathlib import Path

# Get the directory where this constants.py file is located
BASE_DIR = Path(__file__).parent

DEFAULT_PARAMS_DIR = BASE_DIR / "default_params.json"
RESULTS_PATH = BASE_DIR / "results"
EXPERIMENTS_PATH = BASE_DIR / "experiments"
LOG_DIR = BASE_DIR / "logs"
MODEL_MANIFEST_DIR = BASE_DIR / "model_manifest"