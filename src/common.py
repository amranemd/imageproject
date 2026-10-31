import hashlib
import json
from pathlib import Path


def ensure_dirs(*paths):
    for p in paths:
        Path(p).mkdir(parents=True, exist_ok=True)


def read_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def write_json(obj, path):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)


def read_ids(path):
    with open(path, "r") as f:
        return [int(line.strip()) for line in f if line.strip()]


def write_ids(ids, path):
    with open(path, "w") as f:
        f.write("\n".join(str(int(i)) for i in ids) + "\n")


def sha256_file(path, chunk=1 << 20):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            block = f.read(chunk)
            if not block:
                break
            h.update(block)
    return h.hexdigest()


def image_path(images_dir, image_id):
    """COCO file names are the 12-digit image id, e.g. 000000391895.jpg"""
    return Path(images_dir) / f"{int(image_id):012d}.jpg"
