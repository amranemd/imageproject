import csv
import random
from pathlib import Path

from PIL import Image

from .common import read_json, write_json, write_ids, sha256_file, image_path


def _opens(path):
    try:
        with Image.open(path) as im:
            im.verify()
        with Image.open(path) as im:
            im.convert("RGB").load()
        return True
    except Exception:
        return False


def create_splits(valid_ids, out_dir, val_count, seed, test_captions_json,
                  test_images_dir, test_count, audit_csv):
    """Knowledge base = all valid ids. Validation = random subset OF the knowledge base.
    Test = random images from COCO val2017 (never in the knowledge base)."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    kb = sorted(valid_ids)
    val = sorted(random.Random(seed).sample(kb, val_count))

    data = read_json(test_captions_json)
    with_caps = {a["image_id"] for a in data["annotations"] if a["caption"].strip()}
    cands = sorted(i["id"] for i in data["images"] if i["id"] in with_caps)
    random.Random(seed + 1).shuffle(cands)

    kb_set = set(kb)
    test, rejected = [], 0
    for iid in cands:
        if iid in kb_set:
            continue
        if not _opens(image_path(test_images_dir, iid)):
            rejected += 1
            continue
        test.append(iid)
        if len(test) == test_count:
            break
    test = sorted(test)

    # Exact-duplicate leakage check: test image bytes vs knowledge-base image bytes
    kb_hashes = set()
    with open(audit_csv) as f:
        for r in csv.DictReader(f):
            if int(r["image_id"]) in kb_set and r["sha256"]:
                kb_hashes.add(r["sha256"])
    test_hash_overlap = sum(1 for iid in test if sha256_file(image_path(test_images_dir, iid)) in kb_hashes)

    write_ids(kb, out_dir / "knowledge_base_ids.txt")
    write_ids(val, out_dir / "validation_ids.txt")
    write_ids(test, out_dir / "test_ids.txt")

    report = {
        "knowledge_base_images": len(kb),
        "validation_images": len(val),
        "test_images": len(test),
        "validation_is_subset_of_knowledge_base": set(val) <= kb_set,
        "test_ids_overlap_with_knowledge_base": len(set(test) & kb_set),
        "test_exact_duplicate_files_in_knowledge_base": test_hash_overlap,
        "test_candidates_rejected_corrupted": rejected,
        "random_seed": seed,
        "note": "Validation images stay in the knowledge base; during validation, exclude the "
                "query image itself from retrieval. Test images come from COCO val2017.",
    }
    write_json(report, out_dir / "split_report.json")
    print(report)
    return report
