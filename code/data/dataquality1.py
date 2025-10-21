# ============================================
# Parallel Dataset Overlap Analysis - between 2 training dataset by sampling 10k sentences from each
# Measures:
#   1. Lexical Overlap (Jaccard)
#   2. Semantic Overlap (Cosine using LaBSE)
# ============================================

import random
from tqdm import tqdm
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer, util
import torch

# -------- CONFIG --------
new_dataset_file = "/home/krishna/karthika/saamayik-en-sa-translation/data/final_data/combn_train.hi"          # your new dataset (one sentence per line)
existing_dataset_file = "/home/krishna/karthika/data/bpcc/hindi.txt" # existing dataset (one sentence per line)
sample_size = 10000                    # number of samples from each dataset
cosine_threshold = 0.90                # consider pairs > this as overlapping
batch_size = 512                       # for embedding computation

# -------- DATA LOADING (with random sampling) --------
def load_sample(filepath, n_samples):
    with open(filepath, "r", encoding="utf-8") as f:
        lines = [l.strip() for l in f if l.strip()]
    if len(lines) > n_samples:
        lines = random.sample(lines, n_samples)
    print(f"Loaded {len(lines)} lines from {filepath}")
    return lines

new_sents = load_sample(new_dataset_file, sample_size)
existing_sents = load_sample(existing_dataset_file, sample_size)

assert len(new_sents) > 0 and len(existing_sents) > 0, "Files are empty or not found."

# -------- LEXICAL OVERLAP (JACCARD) --------
def jaccard_similarity(set1, set2):
    return len(set1 & set2) / len(set1 | set2) if (set1 or set2) else 0

print("\nComputing lexical (Jaccard) overlap...")
token_sets_new = [set(s.split()) for s in new_sents]
token_sets_existing = [set(s.split()) for s in existing_sents]

lexical_overlaps = []
for s_new in tqdm(token_sets_new, desc="Jaccard overlaps"):
    jaccards = [jaccard_similarity(s_new, s_old) for s_old in token_sets_existing]
    lexical_overlaps.append(max(jaccards))

avg_jaccard = np.mean(lexical_overlaps)
print(f"Average maximum Jaccard overlap: {avg_jaccard:.4f}")

# -------- SEMANTIC OVERLAP (COSINE SIMILARITY) --------
print("\nLoading LaBSE model for semantic similarity...")
device = "cuda" if torch.cuda.is_available() else "cpu"
embed_model = SentenceTransformer("sentence-transformers/LaBSE", device=device)

print("Encoding new dataset sentences...")
emb_new = embed_model.encode(new_sents, batch_size=batch_size, convert_to_tensor=True, normalize_embeddings=True)
print("Encoding existing dataset sentences...")
emb_existing = embed_model.encode(existing_sents, batch_size=batch_size, convert_to_tensor=True, normalize_embeddings=True)

print("\nComputing cosine similarities in batches...")
max_similarities = []
for i in tqdm(range(0, len(emb_new), batch_size), desc="Cosine batches"):
    batch_emb = emb_new[i:i+batch_size]
    cos_sims = util.cos_sim(batch_emb, emb_existing)  # [batch_size, len(existing)]
    max_sim = cos_sims.max(dim=1).values
    max_similarities.extend(max_sim.cpu().numpy())

max_similarities = np.array(max_similarities)
avg_cosine = np.mean(max_similarities)
overlap_ratio = np.mean(max_similarities > cosine_threshold)

print(f"\nAverage maximum cosine similarity: {avg_cosine:.4f}")
print(f"Overlap ratio (cosine > {cosine_threshold}): {overlap_ratio:.4f}")

# -------- SAVE RESULTS --------
df = pd.DataFrame({
    "new_sentence": new_sents,
    "max_cosine_sim": max_similarities,
    "max_jaccard": lexical_overlaps
})
df.to_csv("dataset_overlap_results.tsv", sep="\t", index=False, encoding="utf-8")

print("\n✅ Saved detailed overlap results to 'dataset_overlap_results.tsv'")
print("✅ Done!")
