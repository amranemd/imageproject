import csv
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from transformers import CLIPModel, CLIPProcessor

from .common import read_json, image_path


def _device(requested):
    return "cuda" if requested == "cuda" and torch.cuda.is_available() else "cpu"


def _load(model_name, device):
    model = CLIPModel.from_pretrained(model_name).to(device).eval()
    processor = CLIPProcessor.from_pretrained(model_name)
    return model, processor


class _ImageDataset(Dataset):
    def __init__(self, paths, processor):
        self.paths, self.processor = paths, processor

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        img = Image.open(self.paths[i]).convert("RGB")
        return self.processor(images=img, return_tensors="pt")["pixel_values"][0]


def _normalize(x):
    return x / x.norm(dim=-1, keepdim=True)


@torch.no_grad()
def generate_image_embeddings(images_dir, ids, out_dir, prefix, model_name,
                              batch_size, device, num_workers=2, limit=None):
    device = _device(device)
    ids = list(ids)[:limit] if limit else list(ids)
    model, processor = _load(model_name, device)
    paths = [image_path(images_dir, i) for i in ids]
    loader = DataLoader(_ImageDataset(paths, processor), batch_size=batch_size,
                        num_workers=num_workers, shuffle=False)
    chunks = []
    for pixels in tqdm(loader, desc=f"Image embeddings [{prefix}]"):
        pooled = model.vision_model(pixel_values=pixels.to(device)).pooler_output
        chunks.append(_normalize(model.visual_projection(pooled)).cpu())
    emb = torch.cat(chunks).numpy().astype("float32")
    out_dir = Path(out_dir)
    np.save(out_dir / f"{prefix}_image_embeddings.npy", emb)
    np.save(out_dir / f"{prefix}_image_ids.npy", np.array(ids, dtype="int64"))
    print(prefix, "image embeddings:", emb.shape)
    return emb


@torch.no_grad()
def generate_text_embeddings(captions_json, ids, out_dir, prefix, model_name,
                             batch_size, device, limit=None):
    device = _device(device)
    id_set, order = set(ids), {int(i): n for n, i in enumerate(ids)}
    data = read_json(captions_json)
    items = [(a["image_id"], a["id"], a["caption"].strip()) for a in data["annotations"]
             if a["image_id"] in id_set and a["caption"].strip()]
    items.sort(key=lambda t: (order[t[0]], t[1]))
    if limit:
        items = items[:limit]
    model, processor = _load(model_name, device)
    chunks = []
    bs = max(batch_size, 128)
    for s in tqdm(range(0, len(items), bs), desc=f"Caption embeddings [{prefix}]"):
        texts = [t[2] for t in items[s:s + bs]]
        tok = processor.tokenizer(texts, padding=True, truncation=True, max_length=77,
                                  return_tensors="pt").to(device)
        pooled = model.text_model(input_ids=tok["input_ids"], attention_mask=tok["attention_mask"]).pooler_output
        chunks.append(_normalize(model.text_projection(pooled)).cpu())
    emb = torch.cat(chunks).numpy().astype("float32")
    out_dir = Path(out_dir)
    np.save(out_dir / f"{prefix}_caption_embeddings.npy", emb)
    np.save(out_dir / f"{prefix}_caption_image_ids.npy", np.array([t[0] for t in items], dtype="int64"))
    with open(out_dir / f"{prefix}_caption_meta.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["caption_id", "image_id", "caption"])
        for img_id, cap_id, text in items:
            w.writerow([cap_id, img_id, text])
    print(prefix, "caption embeddings:", emb.shape)
    return emb
