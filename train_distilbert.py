#!/usr/bin/env python3
"""Fine-tune DistilBERT for multi-class harassment detection from a CSV file.

Expected CSV columns:
- text: input text
- label: class label (string or integer-like category)
"""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, precision_recall_fscore_support
from sklearn.model_selection import train_test_split
from transformers import (
    AutoTokenizer,
    DataCollatorWithPadding,
    DistilBertForSequenceClassification,
    Trainer,
    TrainingArguments,
)


@dataclass
class HarassmentDataset:
    encodings: dict
    labels: list[int]

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> dict:
        item = {key: np.array(val[idx]) for key, val in self.encodings.items()}
        item["labels"] = np.array(self.labels[idx])
        return item


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fine-tune DistilBERT on harassment CSV data"
    )
    parser.add_argument("--data", default="data.csv", help="Path to CSV dataset")
    parser.add_argument(
        "--model-name",
        default="distilbert-base-uncased",
        help="Hugging Face model name",
    )
    parser.add_argument("--output-dir", default="distilbert-harassment-model")
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--train-batch-size", type=int, default=16)
    parser.add_argument("--eval-batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=5e-5)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    return parser.parse_args()


def compute_metrics(eval_pred: tuple[np.ndarray, np.ndarray]) -> dict:
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)

    precision, recall, f1_weighted, _ = precision_recall_fscore_support(
        labels, preds, average="weighted", zero_division=0
    )
    f1_macro = f1_score(labels, preds, average="macro", zero_division=0)
    acc = accuracy_score(labels, preds)

    return {
        "accuracy": acc,
        "precision_weighted": precision,
        "recall_weighted": recall,
        "f1_weighted": f1_weighted,
        "f1_macro": f1_macro,
    }


def main() -> None:
    args = parse_args()

    df = pd.read_csv(args.data)
    required_columns = {"text", "label"}
    missing = required_columns - set(df.columns)
    if missing:
        raise ValueError(f"Dataset must contain columns {required_columns}. Missing: {missing}")

    df = df[["text", "label"]].dropna()
    df["text"] = df["text"].astype(str)
    df["label"] = df["label"].astype(str)

    labels = sorted(df["label"].unique().tolist())
    label2id = {label: idx for idx, label in enumerate(labels)}
    id2label = {idx: label for label, idx in label2id.items()}

    train_df, val_df = train_test_split(
        df,
        test_size=args.test_size,
        random_state=args.seed,
        stratify=df["label"],
    )

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)

    train_encodings = tokenizer(
        train_df["text"].tolist(),
        truncation=True,
        padding=False,
        max_length=args.max_length,
    )
    val_encodings = tokenizer(
        val_df["text"].tolist(),
        truncation=True,
        padding=False,
        max_length=args.max_length,
    )

    train_dataset = HarassmentDataset(
        encodings=train_encodings,
        labels=[label2id[label] for label in train_df["label"].tolist()],
    )
    val_dataset = HarassmentDataset(
        encodings=val_encodings,
        labels=[label2id[label] for label in val_df["label"].tolist()],
    )

    model = DistilBertForSequenceClassification.from_pretrained(
        args.model_name,
        num_labels=len(labels),
        id2label={k: v for k, v in id2label.items()},
        label2id=label2id,
    )

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        eval_strategy="epoch",
        save_strategy="epoch",
        logging_strategy="epoch",
        learning_rate=args.learning_rate,
        per_device_train_batch_size=args.train_batch_size,
        per_device_eval_batch_size=args.eval_batch_size,
        num_train_epochs=args.epochs,
        weight_decay=args.weight_decay,
        load_best_model_at_end=True,
        metric_for_best_model="f1_weighted",
        greater_is_better=True,
        report_to="none",
        seed=args.seed,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        tokenizer=tokenizer,
        data_collator=DataCollatorWithPadding(tokenizer=tokenizer),
        compute_metrics=compute_metrics,
    )

    trainer.train()

    metrics = trainer.evaluate()
    print("Validation metrics:")
    for key, value in metrics.items():
        if isinstance(value, (int, float)):
            print(f"  {key}: {value:.4f}")
        else:
            print(f"  {key}: {value}")

    os.makedirs(args.output_dir, exist_ok=True)
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)

    with open(os.path.join(args.output_dir, "label_mapping.json"), "w", encoding="utf-8") as f:
        json.dump({"label2id": label2id, "id2label": id2label}, f, indent=2)

    print(f"Saved fine-tuned model and tokenizer to: {args.output_dir}")


if __name__ == "__main__":
    main()
