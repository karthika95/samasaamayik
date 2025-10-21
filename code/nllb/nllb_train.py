from datasets import Dataset, DatasetDict
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    DataCollatorForSeq2Seq,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
)
import torch

# ========== CONFIG ==========
model_name = "facebook/nllb-200-distilled-1.3B"
src_lang = "san_Deva"
tgt_lang = "hin_Deva"
max_length = 256

# Paths
train_src = "/home/karthika/nmt/data/bpcc_parallel/sanskrit.txt"
train_tgt = "/home/karthika/nmt/data/bpcc_parallel/hindi.txt"

# ========== LOAD DATA ==========
def load_parallel(src_path, tgt_path):
    with open(src_path, "r", encoding="utf-8") as fs, open(tgt_path, "r", encoding="utf-8") as ft:
        src_lines = [l.strip() for l in fs.readlines()]
        tgt_lines = [l.strip() for l in ft.readlines()]
    assert len(src_lines) == len(tgt_lines), "Source and target files must have same number of lines"
    return Dataset.from_dict({"src_text": src_lines, "tgt_text": tgt_lines})

# train_ds = load_parallel(train_src, train_tgt)
# dev_ds = load_parallel(dev_src, dev_tgt)

# dataset = DatasetDict({"train": train_ds, "validation": dev_ds})

# load full data (no separate dev file needed)
full_ds = load_parallel(train_src, train_tgt)

# ========== SPLIT TRAIN/DEV (consistent using a fixed seed) ==========
split = full_ds.train_test_split(
    test_size=0.05,   # e.g. 5% of data for validation
    seed=42,          # fixed seed ensures reproducible split
    shuffle=True
)

dataset = DatasetDict({
    "train": split["train"],
    "validation": split["test"]
})

# ========== TOKENIZER & MODEL ==========
tokenizer = AutoTokenizer.from_pretrained(model_name, src_lang=src_lang, tgt_lang=tgt_lang)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

# ========== PREPROCESS ==========
def preprocess(batch):
    inputs = batch["src_text"]
    targets = batch["tgt_text"]
    model_inputs = tokenizer(
        inputs, text_target=targets,
        max_length=max_length, truncation=True
    )
    return model_inputs

tokenized = dataset.map(preprocess, batched=True, remove_columns=dataset["train"].column_names)

# ========== TRAINING SETUP ==========
data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)

training_args = Seq2SeqTrainingArguments(
    output_dir="/data1/karthika/san_hin_mt/nllb/nllb1.3B-sa-hi",
    per_device_train_batch_size=1,      # adjust based on GPU memory
    gradient_accumulation_steps=16,     # effectively increases batch size
    learning_rate=3e-5,
    num_train_epochs=3,
    fp16=torch.cuda.is_available(),
    eval_strategy="epoch",
    save_strategy="epoch",
    predict_with_generate=True,
    logging_dir="./logs",
    logging_steps=100,
    save_total_limit=2,
    report_to="none",

    # load_best_model_at_end=True,
    # metric_for_best_model="eval_loss",
    # greater_is_better=False,
)

# ========== TRAINER ==========
trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=tokenized["train"],
    eval_dataset=tokenized["validation"],
    tokenizer=tokenizer,
    data_collator=data_collator,
)

# ========== TRAIN ==========
trainer.train()

# ========== SAVE ==========
trainer.save_model("/data1/karthika/san_hin_mt/nllb/nllb1.3B-sa-hi-finetuned")
tokenizer.save_pretrained("/data1/karthika/san_hin_mt/nllb/nllb1.3B-sa-hi-finetuned")
