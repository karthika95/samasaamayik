#!/usr/bin/env python3
"""
Evaluate ByT5 Hindi->Sanskrit translation.
Computes BLEU, chrF++, and WER metrics on a test set.

Usage:
  python byt5_test.py \
    --model_dir /backup/karthika/san_hin_mt/byt5/byt5_hin_sa-bpcc \
    --test_src /home/krishna/karthika/data/mteval/ours/combn_test.hi \
    --test_tgt /home/krishna/karthika/data/mteval/ours/combn_test.sa

    python byt5_test.py \
    --model_dir /backup/karthika/san_hin_mt/byt5/byt5_hin_sa-bpcc \
    --test_src /home/krishna/karthika/data/mteval/flores/hindi.txt \
    --test_tgt /home/krishna/karthika/data/mteval/flores/sanskrit.txt

    python byt5_test.py --model_dir /backup/karthika/san_hin_mt/byt5/byt5_san_hin-bpcc/checkpoint-18996 --test_src /home/krishna/karthika/data/mteval/flores/sanskrit.txt --test_tgt /home/krishna/karthika/data/mteval/flores/hindi.txt
    CUDA_VISIBLE_DEVICES=3 python byt5_test.py --model_dir /backup/karthika/san_hin_mt/byt5/byt5_san_hin-bpcc/checkpoint-18996 --test_src /home/krishna/karthika/saamayik-en-sa-translation/data/final_data/combn_test.sa --test_tgt /home/krishna/karthika/saamayik-en-sa-translation/data/final_data/combn_test.hi
"""

import argparse
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import evaluate
import torch
from tqdm import tqdm  # optional progress bar

# ✅ Define device early
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate ByT5 Hindi→Sanskrit translation")
    parser.add_argument('--model_dir', type=str, required=True, help='Directory of fine-tuned ByT5 model')
    parser.add_argument('--test_src', type=str, required=True, help='Path to test Hindi source file')
    parser.add_argument('--test_tgt', type=str, required=True, help='Path to test Sanskrit target file')
    parser.add_argument('--max_source_length', type=int, default=512)
    parser.add_argument('--max_target_length', type=int, default=1024)
    parser.add_argument('--batch_size', type=int, default=8)
    return parser.parse_args()


def main():
    args = parse_args()

    # ========== Load test data ==========
    with open(args.test_src, 'r', encoding='utf-8') as f:
        src_lines = [line.strip() for line in f if line.strip()]
    with open(args.test_tgt, 'r', encoding='utf-8') as f:
        tgt_lines = [line.strip() for line in f if line.strip()]

    if len(src_lines) != len(tgt_lines):
        raise ValueError(f"Source and target files differ in length: {len(src_lines)} vs {len(tgt_lines)}")

    # ========== Load model & tokenizer ==========
    tokenizer = AutoTokenizer.from_pretrained(args.model_dir)
    model = AutoModelForSeq2SeqLM.from_pretrained(args.model_dir).to(device)
    model.eval()

    # ========== Load metrics ==========
    sacrebleu = evaluate.load('sacrebleu')
    chrf = evaluate.load('chrf')
    wer_metric = evaluate.load('wer')

    # ========== Generate predictions ==========
    preds = []
    batch_size = args.batch_size

    for i in tqdm(range(0, len(src_lines), batch_size), desc="Translating"):
        batch_src = src_lines[i:i + batch_size]

        # Tokenize and move tensors to GPU properly
        inputs = tokenizer(
            batch_src,
            return_tensors='pt',
            padding=True,
            truncation=True,
            max_length=args.max_source_length
        )
        inputs = {k: v.to(device) for k, v in inputs.items()}

        # Generate translations on GPU
        with torch.no_grad():
            outputs = model.generate(**inputs, max_length=args.max_target_length)

        decoded = tokenizer.batch_decode(outputs, skip_special_tokens=True)
        preds.extend([d.strip() for d in decoded])

    refs = [r.strip() for r in tgt_lines]

    # ========== Compute metrics ==========
    bleu = sacrebleu.compute(predictions=preds, references=[[r] for r in refs])
    chrf_score = chrf.compute(predictions=preds, references=refs)
    wer_score = wer_metric.compute(predictions=preds, references=refs)

    # ========== Print results ==========
    print("\n=== Evaluation Results ===")
    print(f"BLEU   : {bleu['score']:.2f}")
    print(f"chrF++ : {chrf_score['score']:.2f}")
    print(f"WER    : {wer_score:.2f}")

    # ========== Optional: Save outputs ==========
    with open("byt5_eval_results.txt", "w", encoding="utf-8") as f:
        for s, r, p in zip(src_lines, refs, preds):
            f.write(f"SOURCE: {s}\nREF: {r}\nPRED: {p}\n\n")
        f.write(f"\nBLEU: {bleu['score']:.2f}\nchrF++: {chrf_score['score']:.2f}\nWER: {wer_score:.2f}\n")


if __name__ == "__main__":
    main()
