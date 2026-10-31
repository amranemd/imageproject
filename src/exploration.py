import re
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .common import read_json, write_json


def _words(text):
    return re.findall(r"[a-z']+", text.lower())


def run_exploration(captions_json, instances_json, ids, reports_dir, figures_dir):
    """Explore ONLY the given image ids (the knowledge base)."""
    ids = set(int(i) for i in ids)
    reports_dir, figures_dir = Path(reports_dir), Path(figures_dir)

    cap = read_json(captions_json)
    caps = defaultdict(list)
    for a in cap["annotations"]:
        if a["image_id"] in ids:
            caps[a["image_id"]].append(a["caption"].strip())
    del cap

    inst = read_json(instances_json)
    cat_name = {c["id"]: c["name"] for c in inst["categories"]}
    cat_super = {c["id"]: c["supercategory"] for c in inst["categories"]}
    obj_count = Counter()
    cats_in_image = defaultdict(set)
    cat_objects = Counter()
    for a in inst["annotations"]:
        iid = a["image_id"]
        if iid in ids:
            obj_count[iid] += 1
            cats_in_image[iid].add(a["category_id"])
            cat_objects[a["category_id"]] += 1
    del inst

    rows = []
    all_caption_words = []
    vocab = Counter()
    for iid in sorted(ids):
        lens = [len(_words(c)) for c in caps[iid]]
        for c in caps[iid]:
            w = _words(c)
            vocab.update(w)
        all_caption_words.extend(lens)
        rows.append({
            "image_id": iid,
            "num_objects": obj_count.get(iid, 0),
            "num_categories": len(cats_in_image.get(iid, ())),
            "num_captions": len(caps[iid]),
            "avg_caption_words": float(np.mean(lens)) if lens else 0.0,
        })
    df = pd.DataFrame(rows)
    df["scene_complexity"] = pd.cut(df["num_objects"], bins=[-1, 0, 3, 8, 10**6],
                                    labels=["no objects", "simple (1-3)", "medium (4-8)", "complex (9+)"])
    df.to_csv(reports_dir / "per_image_stats.csv", index=False)

    # Categories
    img_per_cat = Counter()
    for s in cats_in_image.values():
        img_per_cat.update(s)
    cat_df = pd.DataFrame([{
        "category": cat_name[c], "supercategory": cat_super[c],
        "images_containing": img_per_cat.get(c, 0), "total_objects": cat_objects.get(c, 0)}
        for c in cat_name]).sort_values("images_containing", ascending=False)
    cat_df.to_csv(reports_dir / "category_stats.csv", index=False)
    super_df = cat_df.groupby("supercategory")["total_objects"].sum().sort_values(ascending=False)
    super_df.to_csv(reports_dir / "supercategory_stats.csv")

    # ---- Figures ----
    fig, ax = plt.subplots(figsize=(9, 16))
    d = cat_df.iloc[::-1]
    ax.barh(d["category"], d["images_containing"], color="#3b7dd8")
    ax.set_title("Object categories: number of knowledge-base images containing each")
    ax.set_xlabel("Number of images"); ax.set_ylabel("Category")
    fig.tight_layout(); fig.savefig(figures_dir / "01_category_distribution.png", dpi=150); plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(df["num_objects"].clip(upper=30), bins=range(0, 32), color="#3b7dd8", edgecolor="white")
    ax.set_title("Objects per image (values above 30 grouped at 30)")
    ax.set_xlabel("Number of objects"); ax.set_ylabel("Number of images")
    fig.tight_layout(); fig.savefig(figures_dir / "02_objects_per_image.png", dpi=150); plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(all_caption_words, bins=range(0, max(all_caption_words) + 2), color="#2a9d8f", edgecolor="white")
    ax.set_title("Caption length (all human captions)")
    ax.set_xlabel("Words per caption"); ax.set_ylabel("Number of captions")
    fig.tight_layout(); fig.savefig(figures_dir / "03_caption_length.png", dpi=150); plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    axes[0].scatter(df["num_objects"], df["avg_caption_words"], s=4, alpha=0.25, color="#e76f51")
    axes[0].set_title("Scene complexity vs caption length")
    axes[0].set_xlabel("Objects in image"); axes[0].set_ylabel("Average caption words")
    order = ["no objects", "simple (1-3)", "medium (4-8)", "complex (9+)"]
    counts = df["scene_complexity"].value_counts().reindex(order)
    axes[1].bar(order, counts.values, color="#3b7dd8")
    axes[1].set_title("Images per complexity level"); axes[1].set_ylabel("Number of images")
    fig.tight_layout(); fig.savefig(figures_dir / "04_scene_complexity.png", dpi=150); plt.close(fig)

    corr = float(df["num_objects"].corr(df["avg_caption_words"]))
    summary = {
        "images_analysed": len(df),
        "total_captions": int(df["num_captions"].sum()),
        "objects_per_image": {"mean": float(df.num_objects.mean()), "median": float(df.num_objects.median()),
                              "max": int(df.num_objects.max()),
                              "images_with_zero_objects": int((df.num_objects == 0).sum())},
        "categories_present": int((cat_df["images_containing"] > 0).sum()),
        "top_10_categories_by_images": cat_df.head(10)[["category", "images_containing"]].to_dict("records"),
        "caption_words": {"mean": float(np.mean(all_caption_words)), "median": float(np.median(all_caption_words)),
                          "min": int(np.min(all_caption_words)), "max": int(np.max(all_caption_words))},
        "vocabulary_size": len(vocab),
        "top_30_words": vocab.most_common(30),
        "complexity_counts": {k: int(v) for k, v in counts.items()},
        "correlation_objects_vs_caption_length": corr,
        "note": "Scene complexity is a simple proxy based on the number of annotated objects.",
    }
    write_json(summary, reports_dir / "exploration_summary.json")
    print(summary)
    return summary
