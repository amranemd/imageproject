import csv
import random
from collections import Counter
from pathlib import Path

from PIL import Image
from tqdm import tqdm

from .common import read_json, write_json, sha256_file, image_path


def audit_dataset(images_dir, captions_json, reports_dir, seed, pool_size, keep_count):
    """Audit a seeded random pool of train images and return the clean, unique ids to keep.

    Checks: missing file, missing caption, corrupted image, exact duplicate (SHA-256).
    For duplicates, only the first copy is kept. Original files are never modified.
    """
    data = read_json(captions_json)
    caption_count = Counter()
    for a in data["annotations"]:
        if a["caption"].strip():
            caption_count[a["image_id"]] += 1

    all_ids = sorted(img["id"] for img in data["images"])
    random.Random(seed).shuffle(all_ids)
    candidates = all_ids[:pool_size]

    seen_hash = {}
    rows, valid = [], []
    for iid in tqdm(candidates, desc="Auditing images"):
        path = image_path(images_dir, iid)
        status, digest, dup_of = "valid", "", ""
        if not path.exists():
            status = "missing_file"
        elif caption_count[iid] == 0:
            status = "missing_caption"
        else:
            try:
                with Image.open(path) as im:
                    im.verify()
                with Image.open(path) as im:
                    im.convert("RGB").load()
            except Exception:
                status = "corrupted"
            else:
                digest = sha256_file(path)
                if digest in seen_hash:
                    status, dup_of = "duplicate", seen_hash[digest]
                else:
                    seen_hash[digest] = iid
                    valid.append(iid)
        rows.append({"image_id": iid, "status": status, "sha256": digest, "duplicate_of": dup_of})

    kept = valid[:keep_count]
    kept_set = set(kept)
    for r in rows:
        if r["status"] == "valid" and r["image_id"] not in kept_set:
            r["status"] = "valid_unused_surplus"

    reports_dir = Path(reports_dir)
    with open(reports_dir / "cleaning_audit.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["image_id", "status", "sha256", "duplicate_of"])
        w.writeheader()
        w.writerows(rows)

    counts = Counter(r["status"] for r in rows)
    summary = {
        "images_in_train_json": len(all_ids),
        "audited_pool_size": len(candidates),
        "missing_file": counts["missing_file"],
        "missing_caption": counts["missing_caption"],
        "corrupted": counts["corrupted"],
        "exact_duplicates_removed": counts["duplicate"],
        "valid_unique_in_pool": len(valid),
        "kept_for_knowledge_base": len(kept),
        "target_knowledge_base_size": keep_count,
        "reached_target": len(kept) == keep_count,
        "note": "Duplicates are byte-identical files (SHA-256); first copy kept. "
                "Near-duplicates (resized/recompressed) are checked later with CLIP embeddings.",
    }
    write_json(summary, reports_dir / "cleaning_summary.json")
    print(summary)
    return sorted(kept)
