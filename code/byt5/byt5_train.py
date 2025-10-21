#!/usr/bin/env python3
"""
Train ByT5-base for Hindi -> Sanskrit translation using Hugging Face Transformers and Datasets.

This version automatically loads the dataset from separate source and target training files,
combines them, and splits 5% of the data as validation set using a fixed random seed (42).

Example usage:
  python byt5_train_hin_sa.py \
    --train_src /home/krishna/karthika/saamayik-en-sa-translation/data/final_data/combn_train.hi \
    --train_tgt /home/krishna/karthika/saamayik-en-sa-translation/data/final_data/combn_train.sa \
    --output_dir ./byt5_hin_sa_checkpoint \
    --resume_from_checkpoint /backup/karthika/san_hin_mt/byt5/byt5_hin_sa-bpcc/checkpoint-4749 \
    --num_train_epochs 3

python byt5_train.py \
    --train_src /home/krishna/karthika/data/bpcc/hindi.txt \
    --train_tgt /home/krishna/karthika/data/bpcc/sanskrit.txt \
    --output_dir /backup/karthika/san_hin_mt/byt5/byt5_hin_sa-bpcc \
    --resume_from_checkpoint /backup/karthika/san_hin_mt/byt5/byt5_hin_sa-bpcc/checkpoint-4749 \
    --num_train_epochs 3
    
"""

import argparse
import logging
import os
import pandas as pd
import numpy as np
from typing import List

from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    DataCollatorForSeq2Seq,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
)
import evaluate

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def parse_args():
    parser = argparse.ArgumentParser(description="Train ByT5 on Sanskrit->Hindi translation")
    parser.add_argument("--train_src", type=str, required=True, help="Path to Sanskrit source file")
    parser.add_argument("--train_tgt", type=str, required=True, help="Path to Hindi target file")
    parser.add_argument("--output_dir", type=str, default="/backup/karthika/san_hin_mt/byt5/byt5_san_hin-bpcc")
    parser.add_argument("--model_name", type=str, default="google/byt5-base")
    parser.add_argument('--resume_from_checkpoint', type=str, default=None,
                    help='Path to a checkpoint directory to resume training from.')
    parser.add_argument("--max_source_length", type=int, default=512)
    parser.add_argument("--max_target_length", type=int, default=1024)
    parser.add_argument("--per_device_train_batch_size", type=int, default=4)
    parser.add_argument("--per_device_eval_batch_size", type=int, default=8)
    parser.add_argument("--learning_rate", type=float, default=5e-5)
    parser.add_argument("--num_train_epochs", type=int, default=3)
    parser.add_argument("--save_total_limit", type=int, default=3)
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--overwrite_output_dir", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()

    # Load source and target files
    with open(args.train_src, 'r', encoding='utf-8') as f:
        src_lines = [line.strip() for line in f if line.strip()]
    with open(args.train_tgt, 'r', encoding='utf-8') as f:
        tgt_lines = [line.strip() for line in f if line.strip()]

    if len(src_lines) != len(tgt_lines):
        raise ValueError(f"Source and target files have different number of lines: {len(src_lines)} vs {len(tgt_lines)}")

    df = pd.DataFrame({'sa': src_lines, 'hi': tgt_lines})

    # Convert to HuggingFace Dataset
    full_dataset = Dataset.from_pandas(df)

    # Split into train/validation (95/5)
    split_datasets = full_dataset.train_test_split(test_size=0.05, seed=42)
    train_dataset = split_datasets['train']
    val_dataset = split_datasets['test']

    # Load tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(args.model_name)

    prefix = "translate Sanskrit to Hindi: "

    def preprocess_function(examples):
        inputs = [prefix + s for s in examples['sa']]
        targets = examples['hi']
        model_inputs = tokenizer(inputs, max_length=args.max_source_length, truncation=True)
        labels = tokenizer(text_target=targets, max_length=args.max_target_length, truncation=True)
        model_inputs['labels'] = labels['input_ids']
        return model_inputs

    tokenized_train = train_dataset.map(preprocess_function, batched=True, remove_columns=train_dataset.column_names)
    tokenized_val = val_dataset.map(preprocess_function, batched=True, remove_columns=val_dataset.column_names)

    data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)

    sacrebleu = evaluate.load("sacrebleu")
    chrf = evaluate.load("chrf")

    def postprocess_text(preds: List[str], refs: List[str]):
        preds = [p.strip() for p in preds]
        refs = [[r.strip()] for r in refs]
        return preds, refs

    def compute_metrics(eval_pred):
        preds, labels = eval_pred
        if isinstance(preds, tuple):
            preds = preds[0]
        decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
        labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
        decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
        decoded_preds, decoded_labels = postprocess_text(decoded_preds, decoded_labels)

        bleu = sacrebleu.compute(predictions=decoded_preds, references=decoded_labels)
        chrf_score = chrf.compute(predictions=decoded_preds, references=[r[0] for r in decoded_labels])

        result = {"bleu": bleu['score'], "chrf": chrf_score['score']}
        prediction_lens = [np.count_nonzero(tokenizer.encode(p)) for p in decoded_preds]
        result['gen_len'] = float(np.mean(prediction_lens))
        return result

    training_args = Seq2SeqTrainingArguments(
        output_dir=args.output_dir,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        learning_rate=args.learning_rate,
        num_train_epochs=args.num_train_epochs,
        weight_decay=0.01,
        predict_with_generate=True,
        fp16=args.fp16,
        save_total_limit=args.save_total_limit,
        logging_dir=os.path.join(args.output_dir, "logs"),
        logging_strategy="steps",
        logging_steps=200,
        load_best_model_at_end=True,
        metric_for_best_model="bleu",
        greater_is_better=True,
        remove_unused_columns=False,
    )

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train,
        eval_dataset=tokenized_val,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    # trainer.train()
    trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)


if __name__ == "__main__":
    main()