import csv
from pathlib import Path

import faiss
import numpy as np

from .common import write_json


def build_faiss_indexes(embeddings_dir, index_dir, near_dup_threshold=0.95):
    emb, idx = Path(embeddings_dir), Path(index_dir)
    img = np.load(emb / "kb_image_embeddings.npy")
    img_ids = np.load(emb / "kb_image_ids.npy")
    txt = np.load(emb / "kb_caption_embeddings.npy")
    txt_img_ids = np.load(emb / "kb_caption_image_ids.npy")
    dim = img.shape[1]

    # Vectors are L2-normalised, so inner product == cosine similarity.
    image_index = faiss.IndexFlatIP(dim); image_index.add(img)
    caption_index = faiss.IndexFlatIP(dim); caption_index.add(txt)
    faiss.write_index(image_index, str(idx / "kb_image.index"))
    faiss.write_index(caption_index, str(idx / "kb_caption.index"))
    np.save(idx / "kb_image_ids.npy", img_ids)
    np.save(idx / "kb_caption_image_ids.npy", txt_img_ids)

    # Sanity checks
    n = min(100, len(img))
    _, I = image_index.search(img[:n], 1)
    self_retrieval = float((img_ids[I[:, 0]] == img_ids[:n]).mean())
    m = min(500, len(img))
    _, I = caption_index.search(img[:m], 5)
    own_caption_recall5 = float(np.mean([(txt_img_ids[I[r]] == img_ids[r]).any() for r in range(m)]))

    # Near-duplicates inside the knowledge base
    D, I = image_index.search(img, 3)
    pairs = set()
    for r in range(len(img)):
        for s, j in zip(D[r], I[r]):
            if j != r and s >= near_dup_threshold:
                pairs.add((min(int(img_ids[r]), int(img_ids[j])), max(int(img_ids[r]), int(img_ids[j])), round(float(s), 4)))
                break
    with open(idx / "near_duplicates_in_kb.csv", "w", newline="") as f:
        w = csv.writer(f); w.writerow(["image_id_a", "image_id_b", "cosine"]); w.writerows(sorted(pairs))

    # Test vs knowledge base (leakage check)
    test_flagged, test_max = None, None
    test_path = emb / "test_image_embeddings.npy"
    if test_path.exists():
        t = np.load(test_path); t_ids = np.load(emb / "test_image_ids.npy")
        D, I = image_index.search(t, 1)
        test_max = float(D.max())
        flagged = [(int(t_ids[r]), int(img_ids[I[r, 0]]), round(float(D[r, 0]), 4))
                   for r in range(len(t)) if D[r, 0] >= near_dup_threshold]
        test_flagged = len(flagged)
        with open(idx / "near_duplicates_test_vs_kb.csv", "w", newline="") as f:
            w = csv.writer(f); w.writerow(["test_image_id", "kb_image_id", "cosine"]); w.writerows(flagged)

    report = {
        "embedding_model": "openai/clip-vit-base-patch32 (L2-normalised, dim %d)" % dim,
        "index_type": "faiss.IndexFlatIP (exact cosine search)",
        "image_vectors": int(image_index.ntotal),
        "caption_vectors": int(caption_index.ntotal),
        "sanity_image_finds_itself_top1": self_retrieval,
        "sanity_image_to_own_caption_recall_at_5": own_caption_recall5,
        "near_duplicate_threshold": near_dup_threshold,
        "near_duplicate_pairs_in_knowledge_base": len(pairs),
        "test_images_too_similar_to_knowledge_base": test_flagged,
        "test_highest_similarity_to_knowledge_base": test_max,
    }
    write_json(report, idx / "index_report.json")
    print(report)
    return report
