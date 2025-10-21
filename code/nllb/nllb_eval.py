from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch
import evaluate

# ========== CONFIG ==========
# model_dir = "/data1/karthika/san_hin_mt/nllb/nllb1.3B-hi-sa/checkpoint-2674"
model_dir = "/data1/karthika/san_hin_mt/nllb/nllb1.3B-sa-hi-finetuned-bpcc/checkpoint-2376"
test_src = "/home/karthika/nmt/eval_data/in22/sanskrit.txt"
test_tgt = "/home/karthika/nmt/eval_data/in22/hindi.txt"
src_lang = "san_Deva"
tgt_lang = "hin_Deva"
max_length = 256

# ========== LOAD MODEL & TOKENIZER ==========
device = "cuda" if torch.cuda.is_available() else "cpu"
tokenizer = AutoTokenizer.from_pretrained(model_dir, src_lang=src_lang, tgt_lang=tgt_lang, local_files_only=True)
model = AutoModelForSeq2SeqLM.from_pretrained(model_dir, local_files_only=True).to(device)

# ========== LOAD TEST DATA ==========
def load_parallel(src_path, tgt_path):
    with open(src_path, "r", encoding="utf-8") as fs, open(tgt_path, "r", encoding="utf-8") as ft:
        src_lines = [l.strip() for l in fs.readlines()]
        tgt_lines = [l.strip() for l in ft.readlines()]
    assert len(src_lines) == len(tgt_lines), "Source and target files must have same number of lines"
    return Dataset.from_dict({"src_text": src_lines, "tgt_text": tgt_lines})

test_ds = load_parallel(test_src, test_tgt)

# ========== GENERATE TRANSLATIONS ==========
def generate_translation(batch):
    inputs = tokenizer(
        batch["src_text"], return_tensors="pt",
        padding=True,
        truncation=True, max_length=max_length
    ).to(device)

    tokenizer.src_lang = src_lang
    tokenizer.tgt_lang = tgt_lang
    forced_bos_id = tokenizer.convert_tokens_to_ids(tgt_lang)
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            # forced_bos_token_id=tokenizer.lang_code_to_id[tgt_lang],
            forced_bos_token_id=forced_bos_id,
            max_length=max_length
        )

    preds = tokenizer.batch_decode(outputs, skip_special_tokens=True)
    return {"predictions": preds}

results = test_ds.map(generate_translation, batched=True, batch_size=8)

# ========== LOAD METRICS ==========
bleu = evaluate.load("sacrebleu")
chrf = evaluate.load("chrf")
wer = evaluate.load("wer")

# sacrebleu expects list of list for references
refs = [[ref] for ref in test_ds["tgt_text"]]
preds = results["predictions"]

# ========== COMPUTE METRICS ==========
bleu_score = bleu.compute(predictions=preds, references=refs)["score"]
chrf_score = chrf.compute(predictions=preds, references=refs)["score"]
wer_score = wer.compute(predictions=preds, references=test_ds["tgt_text"])

# ========== DISPLAY RESULTS ==========
print("\n Evaluation Results:")
print(f"  BLEU  : {bleu_score:.2f}")
print(f"  chrF++: {chrf_score:.2f}")
print(f"  WER   : {wer_score:.2f}")

# ========== (OPTIONAL) SAVE OUTPUTS ==========
with open("test_predictions_metrics.txt", "w", encoding="utf-8") as f:
    for src, ref, pred in zip(test_ds["src_text"], test_ds["tgt_text"], preds):
        f.write(f"SOURCE: {src}\nREFERENCE: {ref}\nPREDICTION: {pred}\n\n")

    f.write("\n=== METRICS SUMMARY ===\n")
    f.write(f"BLEU  : {bleu_score:.2f}\n")
    f.write(f"chrF++: {chrf_score:.2f}\n")
    f.write(f"WER   : {wer_score:.2f}\n")
