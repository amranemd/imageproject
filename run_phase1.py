import argparse

import config
from src.common import ensure_dirs, read_ids, read_json, write_json
from src.cleaning import audit_dataset
from src.splitting import create_splits
from src.exploration import run_exploration

SPLIT_DIR = config.PROCESSED_DIR / "splits"


def main():
    parser = argparse.ArgumentParser(description="MS-COCO multimodal RAG - Phase 1")
    parser.add_argument("--step", choices=["clean", "split", "explore", "embed", "index", "all"], default="all")
    args = parser.parse_args()
    ensure_dirs(config.OUTPUT_DIR, config.PROCESSED_DIR, config.REPORTS_DIR, config.FIGURES_DIR,
                config.EMBEDDINGS_DIR, config.INDEX_DIR, SPLIT_DIR)
    run = lambda s: args.step in (s, "all")

    if run("clean"):
        kept = audit_dataset(config.TRAIN_IMAGES_DIR, config.TRAIN_CAPTIONS_JSON, config.REPORTS_DIR,
                             config.RANDOM_SEED, config.CANDIDATE_POOL, config.KB_COUNT)
        write_json({"valid_unique_image_count": len(kept), "valid_unique_image_ids": kept},
                   config.PROCESSED_DIR / "valid_image_ids.json")

    if run("split"):
        valid = read_json(config.PROCESSED_DIR / "valid_image_ids.json")["valid_unique_image_ids"]
        create_splits(valid, SPLIT_DIR, config.VALIDATION_COUNT, config.RANDOM_SEED,
                      config.TEST_CAPTIONS_JSON, config.TEST_IMAGES_DIR, config.TEST_COUNT,
                      config.REPORTS_DIR / "cleaning_audit.csv")

    if run("explore"):
        run_exploration(config.TRAIN_CAPTIONS_JSON, config.TRAIN_INSTANCES_JSON,
                        read_ids(SPLIT_DIR / "knowledge_base_ids.txt"),
                        config.REPORTS_DIR, config.FIGURES_DIR)

    if run("embed"):
        from src.embeddings import generate_image_embeddings, generate_text_embeddings
        kb = read_ids(SPLIT_DIR / "knowledge_base_ids.txt")
        test = read_ids(SPLIT_DIR / "test_ids.txt")
        generate_image_embeddings(config.TRAIN_IMAGES_DIR, kb, config.EMBEDDINGS_DIR, "kb",
                                  config.CLIP_MODEL_NAME, config.BATCH_SIZE, config.DEVICE,
                                  config.NUM_WORKERS, config.EMBEDDING_IMAGE_LIMIT)
        generate_text_embeddings(config.TRAIN_CAPTIONS_JSON, kb, config.EMBEDDINGS_DIR, "kb",
                                 config.CLIP_MODEL_NAME, config.BATCH_SIZE, config.DEVICE)
        generate_image_embeddings(config.TEST_IMAGES_DIR, test, config.EMBEDDINGS_DIR, "test",
                                  config.CLIP_MODEL_NAME, config.BATCH_SIZE, config.DEVICE,
                                  config.NUM_WORKERS)

    if run("index"):
        from src.knowledge_base import build_faiss_indexes
        build_faiss_indexes(config.EMBEDDINGS_DIR, config.INDEX_DIR, config.NEAR_DUPLICATE_THRESHOLD)

    print("Step completed:", args.step)


if __name__ == "__main__":
    main()
