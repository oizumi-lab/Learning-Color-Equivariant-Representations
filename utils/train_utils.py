from typing import List, Dict, Tuple
import constants as c
import json

def get_default_params():
    with open(c.DEFAULT_PARAMS_DIR) as f:
        return json.load(f)


def add_to_manifest(params: Dict, file_path: str) -> None:
    with open(file_path, 'a') as f:
        json.dump(params, f)
    