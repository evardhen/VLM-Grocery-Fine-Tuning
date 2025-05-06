import json
from pathlib import Path
from typing import Any, List, Dict

def add_question_ids(
    json_path: List[str | Path],
    key: str = "question_id",
    overwrite: bool = True,
):
    """
    Load *json_path* (a list of dicts), append a sequential ID field, and
    return the updated list.
    """
    idx = 0
    for path in json_path:
        path = Path(path)

        with path.open() as f:
            data: List[Dict[str, Any]] = json.load(f)

        for entry in data:
            entry[key] = idx
            idx += 1

        if overwrite:
            path.write_text(json.dumps(data, indent=4))

if __name__ == "__main__":
    path_list = ["data/eval.json", "data/zeki_grocery_train.json", "data/fruits_and_vegs_grocery.json", "data/freiburg_grocery.json"]
    add_question_ids(path_list)