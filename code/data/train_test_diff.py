# ============================================
# Novelty / Overlap of Test Set vs Training Set
# ============================================

from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm
import numpy as np
import torch

# ========== CONFIG ==========
train_file = "/home/krishna/karthika/saamayik-en-sa-translation/data/final_data/combn_train.hi"
test_file = "/home/krishna/karthika/data/mteval/IN22/hindi.txt"
sample_size_train = 10000   # optional, for speed — use all if feasible
batch_size = 128

# ========== LOAD DATA ==========
with open(train_file, "r", encoding="utf-8") as f:
    train_sents = [line.strip() for line in f if line.strip()]

with open(test_file, "r", encoding="utf-8") as f:
    test_sents = [line.strip() for line in f if line.strip()]

print(f"Training set size: {len(train_sents)}")
print(f"Test set size: {len(test_sents)}")

# Optional: downsample training for compute efficiency
if len(train_sents) > sample_size_train:
    import random
    random.seed(42)
    train_sents = random.sample(train_sents, sample_size_train)
    print(f"Sampled {len(train_sents)} training sentences for comparison.")

# ========== 1️⃣ Lexical Jaccard Overlap ==========
def jaccard_similarity(a, b):
    set_a, set_b = set(a.split()), set(b.split())
    if not set_a or not set_b:
        return 0.0
    return len(set_a & set_b) / len(set_a | set_b)

print("\nComputing lexical (Jaccard) overlap...")
jaccard_max = []
for test_sent in tqdm(test_sents, desc="Test sentences"):
    sims = [jaccard_similarity(test_sent, train_sent) for train_sent in train_sents]
    jaccard_max.append(max(sims))

avg_jaccard = np.mean(jaccard_max)
print(f"Average maximum Jaccard overlap: {avg_jaccard:.4f}")

# ========== 2️⃣ Semantic Cosine Similarity ==========
print("\nLoading LaBSE model for semantic similarity...")
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = SentenceTransformer('sentence-transformers/LaBSE').to(device)

def encode_sentences(sentences):
    embeddings = []
    for i in range(0, len(sentences), batch_size):
        batch = sentences[i:i+batch_size]
        emb = model.encode(batch, convert_to_tensor=True, show_progress_bar=False)
        embeddings.append(emb)
    return torch.cat(embeddings)

print("Encoding sentences...")
test_emb = encode_sentences(test_sents)
train_emb = encode_sentences(train_sents)

print("\nComputing cosine similarities in batches...")
cosine_max = []
for i in tqdm(range(0, len(test_emb), batch_size), desc="Cosine batches"):
    batch = test_emb[i:i+batch_size]
    sims = cosine_similarity(batch.cpu(), train_emb.cpu())
    cosine_max.extend(np.max(sims, axis=1))

avg_cosine = np.mean(cosine_max)
overlap_ratio = np.mean(np.array(cosine_max) > 0.9)

print(f"\nAverage maximum cosine similarity: {avg_cosine:.4f}")
print(f"Overlap ratio (cosine > 0.9): {overlap_ratio:.4f}")
