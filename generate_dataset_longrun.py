#!/usr/bin/env python3
"""
generate_dataset_longrun.py

RunPod-only long-running dataset generator with incremental FAISS deduping.

Requirements:
  pip install python-dotenv requests tqdm sentence-transformers faiss-cpu

Usage (pilot):
  python generate_dataset_longrun.py --target-size 200 --batch-seeds 5 --per-seed 4

Usage (full):
  python generate_dataset_longrun.py --target-size 400000 --batch-seeds 50 --per-seed 8
"""

import os
import json
import time
import argparse
import requests
from tqdm import tqdm
from pathlib import Path
from dotenv import load_dotenv

# embeddings
from sentence_transformers import SentenceTransformer
import numpy as np
import faiss

# Load env
load_dotenv()
RUNPOD_API_URL = os.getenv("RUNPOD_API_URL")
RUNPOD_API_KEY = os.getenv("RUNPOD_API_KEY")
if not RUNPOD_API_URL or not RUNPOD_API_KEY:
    raise SystemExit("Please set RUNPOD_API_URL and RUNPOD_API_KEY in .env")

# Files and checkpoints
DATA_DIR = Path("data")
DATA_DIR.mkdir(exist_ok=True)
OUTPUT_FILE = DATA_DIR / "rafiki_dataset.jsonl"
INDEX_FILE = DATA_DIR / "rafiki_faiss.index"
METADATA_FILE = DATA_DIR / "rafiki_metadata.jsonl"

# Embedding model
EMBED_MODEL_NAME = "all-MiniLM-L6-v2"  # fast + good
embedder = SentenceTransformer(EMBED_MODEL_NAME)
EMB_DIM = embedder.get_sentence_embedding_dimension()

# Default prompting function (customize)
def make_prompt_for_style(seed_instruction, style=None, num_pairs=5):
    # Ask the model to output JSONL lines: {"instruction": "...", "response": "..."}
    style_text = f" as a {style} coach" if style else ""
    prompt = (
        f"You are an expert coach{style_text}. For the user prompt below, generate {num_pairs} "
        "distinct coaching question-and-answer pairs inspired by the prompt. Each generated line must be valid JSON with fields: "
        "'instruction' and 'response'. Keep answers concise (1-3 sentences), practical and empathetic.\n\n"
        f"Seed: {seed_instruction}\n\nOutput JSONL lines only, one per line."
    )
    return prompt

# Call RunPod
def call_runpod(prompt, max_tokens=512, temperature=0.8):
    headers = {
        "Authorization": f"Bearer {RUNPOD_API_KEY}",
        "Content-Type": "application/json",
    }
    payload = {
        "input": {
            "prompt": prompt,
            "max_tokens": max_tokens,
            "temperature": temperature
        }
    }
    r = requests.post(RUNPOD_API_URL, json=payload, headers=headers, timeout=300)
    r.raise_for_status()
    return r.json()

# Parsing helper: extract JSON objects from response text
def parse_jsonl_from_text(text):
    lines = []
    for raw in text.splitlines():
        raw = raw.strip()
        if not raw:
            continue
        # sometimes the model returns bullet points or JSON with single quotes — be generous
        try:
            obj = json.loads(raw)
            if "instruction" in obj and "response" in obj:
                lines.append(obj)
                continue
        except:
            pass
        # try to find JSON substring
        start = raw.find("{")
        end = raw.rfind("}")
        if start != -1 and end != -1 and end > start:
            snippet = raw[start:end+1]
            try:
                obj = json.loads(snippet)
                if "instruction" in obj and "response" in obj:
                    lines.append(obj)
                    continue
            except:
                pass
        # fallback: skip noisy lines
    return lines

# FAISS util functions
def create_faiss_index(dim, index_path=None):
    index = faiss.IndexFlatIP(dim)  # inner product (cosine after normalization)
    if index_path and Path(index_path).exists():
        faiss.read_index(str(index_path))  # not used—prefer loading externally
    return index

def normalize_embeddings(embs):
    # L2-normalize for cosine similarity via inner product
    norms = np.linalg.norm(embs, axis=1, keepdims=True)
    norms[norms == 0] = 1e-6
    return embs / norms

def save_index(index, path):
    faiss.write_index(index, str(path))

def load_index(path):
    return faiss.read_index(str(path))

def main(args):
    # Load seeds
    seeds_file = Path(args.seeds_file)
    if not seeds_file.exists():
        raise SystemExit(f"Seeds file not found: {seeds_file}")
    with open(seeds_file, "r", encoding="utf-8") as f:
        seeds = [json.loads(line)["instruction"] for line in f if line.strip()]

    # Prepare or load existing index & metadata
    if INDEX_FILE.exists() and METADATA_FILE.exists():
        print("Loading existing FAISS index and metadata (resume mode)...")
        index = load_index(INDEX_FILE)
        metadata = []
        with open(METADATA_FILE, "r", encoding="utf-8") as mf:
            for line in mf:
                metadata.append(json.loads(line))
        print(f"Loaded {len(metadata)} existing entries.")
        # verify dimension matches
        if index.d != EMB_DIM:
            print("Warning: FAISS index dim mismatch; rebuilding index.")
            index = faiss.IndexFlatIP(EMB_DIM)
            metadata = []
    else:
        index = faiss.IndexFlatIP(EMB_DIM)
        metadata = []

    # If dataset file exists, we will append unique entries
    existing_count = len(metadata)
    print(f"Starting: existing unique items = {existing_count}")

    # Loop until we reach target
    pbar = tqdm(total=args.target_size, desc="Unique items")
    pbar.update(existing_count)

    # We will iterate seeds in a shuffled continuous loop; batch multiple seeds per call
    seed_idx = 0
    attempt = 0
    saved_since = 0

    # store local cache of embeddings to avoid repeated compute when checkpointing
    # metadata list contains dicts with 'instruction' and 'response'
    # For FAISS we need embeddings of metadata. If index already has vectors, we rely on it.
    if existing_count > 0:
        # nothing more needed; index already contains vectors
        pass

    while len(metadata) < args.target_size:
        attempt += 1
        # prepare a batch of seeds
        batch_seeds = []
        for i in range(args.batch_seeds):
            batch_seeds.append(seeds[seed_idx % len(seeds)])
            seed_idx += 1

        # For each seed, call RunPod in a loop (or combine seeds into one prompt)
        for seed in batch_seeds:
            prompt = make_prompt_for_style(seed, style=None, num_pairs=args.per_seed)
            try:
                resp_json = call_runpod(prompt, max_tokens=args.max_tokens, temperature=args.temperature)
            except Exception as e:
                print("RunPod call failed:", e)
                time.sleep(5)
                continue

            # Try to extract textual output from response. RunPod responses vary by model.
            # Common keys: 'output' or 'output'->'text' or 'result'
            text = None
            if isinstance(resp_json, dict):
                # Inspect likely fields
                if "output" in resp_json:
                    out = resp_json["output"]
                    if isinstance(out, dict) and "text" in out:
                        text = out["text"]
                    elif isinstance(out, str):
                        text = out
                if text is None:
                    # some runpods return {'id':..., 'output': '...'}
                    # or {'result': '...'}
                    if "result" in resp_json and isinstance(resp_json["result"], str):
                        text = resp_json["result"]
            if text is None:
                # fallback: stringify entire json
                text = json.dumps(resp_json)

            new_items = parse_jsonl_from_text(text)
            if not new_items:
                # Try a second attempt with a simpler parse: treat lines as "Q: ... A: ..." pairs
                # (left out for brevity)
                continue

            # For each new item, compute embedding of instruction and check similarity
            instructions = [it["instruction"].strip() for it in new_items]
            if not instructions:
                continue
            embs = embedder.encode(instructions, convert_to_numpy=True, show_progress_bar=False)
            embs = normalize_embeddings(embs).astype("float32")

            if index.ntotal > 0:
                # search for nearest neighbors
                D, I = index.search(embs, k=1)  # inner product -> cosine since normalized
                similarities = D[:, 0]
                is_unique = similarities < args.similarity_threshold
            else:
                is_unique = np.array([True] * len(instructions))

            # Add unique ones
            unique_count = 0
            for i, it in enumerate(new_items):
                if is_unique[i]:
                    # append to metadata + index
                    metadata.append({"instruction": it["instruction"].strip(), "response": it["response"].strip()})
                    index.add(np.expand_dims(embs[i], axis=0))
                    unique_count += 1
                else:
                    # duplicate, skip
                    pass

            pbar.update(unique_count)
            saved_since += unique_count

            # Periodic checkpoint save
            if saved_since >= args.checkpoint_every:
                print(f"\nCheckpointing: saving {len(metadata)} items and FAISS index...")
                # append to OUTPUT_FILE and METADATA_FILE (append incremental)
                with open(METADATA_FILE, "a", encoding="utf-8") as mf, open(OUTPUT_FILE, "a", encoding="utf-8") as of:
                    # write only the last saved_since items
                    for item in metadata[-saved_since:]:
                        mf.write(json.dumps(item, ensure_ascii=False) + "\n")
                        of.write(json.dumps(item, ensure_ascii=False) + "\n")
                save_index(index, INDEX_FILE)
                saved_since = 0

            # quick stop if we reached target
            if len(metadata) >= args.target_size:
                break

        # slight backoff to avoid throttling
        time.sleep(args.per_call_delay)

    # Final save of remaining items not yet checkpointed
    if saved_since > 0:
        print("Final checkpointing...")
        with open(METADATA_FILE, "a", encoding="utf-8") as mf, open(OUTPUT_FILE, "a", encoding="utf-8") as of:
            for item in metadata[-saved_since:]:
                mf.write(json.dumps(item, ensure_ascii=False) + "\n")
                of.write(json.dumps(item, ensure_ascii=False) + "\n")
        save_index(index, INDEX_FILE)

    pbar.close()
    print(f"\nDONE. Total unique items: {len(metadata)}. Saved to {OUTPUT_FILE}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds-file", type=str, default="data/seeds.jsonl")
    parser.add_argument("--target-size", type=int, default=400000)
    parser.add_argument("--batch-seeds", type=int, default=20,
                        help="how many seeds to process per outer loop (affects RunPod calls)")
    parser.add_argument("--per-seed", type=int, default=5,
                        help="how many pairs requested per seed prompt")
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--per-call-delay", type=float, default=0.5,
                        help="seconds to sleep between API calls to avoid throttling")
    parser.add_argument("--similarity-threshold", type=float, default=0.87,
                        help="cosine similarity threshold to treat as duplicate (0-1). Lower => more strict uniqueness")
    parser.add_argument("--checkpoint-every", type=int, default=5000,
                        help="save to disk after this many new unique items")
    args = parser.parse_args()
    main(args)
