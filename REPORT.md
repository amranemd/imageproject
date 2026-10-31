# Phase 1 Report: Data Exploration and Preprocessing

Dataset: MS-COCO 2017 (train2017 for the knowledge base, val2017 for the test set). Random seed: 42.

## 1. Data cleaning

| Check | Count |
|---|---|
| Images in train2017 annotation file | 118,287 |
| Images audited (seeded random pool) | 10,500 |
| Missing image files | 0 |
| Images without captions | 0 |
| Corrupted images | 0 |
| Exact duplicates removed (SHA-256) | 0 |
| Valid unique images in the pool | 10,500 |
| Kept for the knowledge base | 10,000 |

No problems were found in the audited sample. This is expected, since COCO is a curated dataset,
and it shows the checks ran rather than that they were skipped. The 500 surplus valid images were not used.

## 2. Data splits

| Split | Images | Source |
|---|---|---|
| Knowledge base | 10,000 | Random sample of train2017 |
| Validation | 100 | Random subset of the 10,000 |
| Test | 100 | Random sample of val2017 |

- The validation set is a subset of the knowledge base (checked).
- No test image id belongs to the knowledge base, and no test file is byte-identical to a knowledge-base file.
- Split files: `outputs/processed/splits/` (`knowledge_base_ids.txt`, `validation_ids.txt`, `test_ids.txt`).

## 3. Exploration of the 10,000 knowledge-base images

### Object categories
All 80 COCO categories appear. The most common by number of images containing them:
person (5,382), chair (1,109), car (1,041), dining table (1,034), cup (822),
bottle (714), bowl (626), handbag (599), truck (529), bench (482).
`person` is in about 54% of the images, far ahead of any other category, so the knowledge base is
dominated by scenes with people.

### Objects per image
Mean 7.34, median 4, maximum 66. Only 80 images (0.8%) have no annotated object.
The mean is well above the median, so the distribution is skewed to the right: most images contain a few
objects, and a smaller group of crowded scenes pulls the average up.

### Caption length
The 10,000 images have 50,024 captions (about 5 per image). Captions have a mean of 10.47 words,
a median of 10, a minimum of 6 and a maximum of 49. The vocabulary has 9,732 distinct words.
The most frequent words are mostly function words (a, on, of, the, in, with), followed by content
words such as man, sitting, two, people, standing, white, woman, table and street.
Captions mostly describe people, their posture (sitting, standing, holding) and everyday places.

### Scene complexity
Complexity is a proxy based on the number of objects:

| Level | Objects | Images |
|---|---|---|
| No objects | 0 | 80 (0.8%) |
| Simple | 1 to 3 | 4,184 (41.8%) |
| Medium | 4 to 8 | 2,766 (27.7%) |
| Complex | 9 or more | 2,970 (29.7%) |

The correlation between the number of objects and the average caption length is 0.09, which is very weak.
Annotators write short captions of similar length whether the scene has 2 objects or 30, so they summarise
the main idea of a scene and do not list everything in it. For retrieval, this suggests that a caption
describes the main subject, and that complex scenes will be described by only part of their content.

## 4. Embeddings and index

| Item | Value |
|---|---|
| Model | CLIP ViT-B/32 (`openai/clip-vit-base-patch32`) |
| Vector size | 512, L2-normalised |
| Knowledge-base image embeddings | 10,000 |
| Knowledge-base caption embeddings | 50,024 |
| Test image embeddings | 100 |
| Index | FAISS `IndexFlatIP` (inner product on normalised vectors equals cosine similarity) |

Sanity checks (from `outputs/faiss/index_report.json`):
- Each of the first 100 knowledge-base images finds itself as its top match: 100%.
- For 500 knowledge-base images, at least one of their own captions appears in the top 5
  caption results for 64.8% of them. This is a quick check, not a formal evaluation.

## 5. Near-duplicate check

Using CLIP cosine similarity with a threshold of 0.95:
- 34 pairs of near-similar images were found inside the knowledge base (listed in
  `outputs/faiss/near_duplicates_in_kb.csv`). They were kept, because similar scenes are normal in COCO.
- 1 test image has a similarity of 0.954 to a knowledge-base image (the highest among the 100 test images).
  Manual check: Manual check: the two images were compared visually. They are different photographs of a very similar scene (same animals, environment and postures), probably taken in the same session. The test image was kept, since it is 1 of 100 test images, and this is noted as a limitation: results on this image may be slightly optimistic.

## 6. Assumptions to confirm

- Validation is drawn from inside the 10,000 images (as written in the brief), so the knowledge base keeps 10,000 images.
- The test set has 100 images.
