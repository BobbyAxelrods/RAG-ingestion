import os
from typing import Iterable, List


def discover_json_files(path: str, pattern_suffix: str = "_extraction.json") -> List[str]:
    files: List[str] = []
    if os.path.isfile(path):
        if path.endswith(".json"):
            files.append(path)
        return files
    for root, _, filenames in os.walk(path):
        for name in filenames:
            if name.endswith(pattern_suffix):
                files.append(os.path.join(root, name))
    return files