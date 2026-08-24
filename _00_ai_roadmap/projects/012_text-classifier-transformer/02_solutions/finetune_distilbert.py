"""Fine-tune DistilBERT on Project 011's human-vs-AI text dataset.

Run:
    pip install transformers datasets
    python finetune_distilbert.py
"""
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset
from transformers import (
    AutoModelForSequenceClassification, AutoTokenizer,
    Trainer, TrainingArguments,
)

DATA_PATH = (Path(__file__).parent / "../../011_text-classifier-ai-detector/02_solutions/data/human_vs_ai.csv").resolve()
MODEL_NAME = "distilbert-base-uncased"


class TextDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len=64):
        self.encodings = tokenizer(list(texts), truncation=True, padding=True, max_length=max_len)
        self.labels = list(labels)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        item = {k: torch.tensor(v[idx]) for k, v in self.encodings.items()}
        item["labels"] = torch.tensor(self.labels[idx])
        return item


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=1)
    print(classification_report(labels, preds))
    return {"accuracy": (preds == labels).mean()}


def stress_test(model, tokenizer):
    novel_texts = [
        "honestly not sure how i feel about all this ai stuff, kinda weird ngl",
        "In summary, this topic warrants further investigation and analysis by researchers.",
        "The weather today is nice.",
        "This is a great product I really enjoyed using it every day.",
    ]
    model.eval()
    print("\n=== Stress test (same sentences as Project 011) ===")
    for text in novel_texts:
        inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
        with torch.no_grad():
            logits = model(**inputs).logits
        pred = logits.argmax(dim=-1).item()
        print(f"{text!r} -> predicted {'AI' if pred == 1 else 'human'}")


def main():
    df = pd.read_csv(DATA_PATH)
    X_train, X_test, y_train, y_test = train_test_split(
        df["text"], df["label"], test_size=0.2, stratify=df["label"], random_state=0
    )

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    train_ds = TextDataset(X_train, y_train, tokenizer)
    test_ds = TextDataset(X_test, y_test, tokenizer)

    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=2)

    args = TrainingArguments(
        output_dir="./results",
        num_train_epochs=3,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        eval_strategy="epoch",
        logging_steps=10,
        report_to=[],
    )

    trainer = Trainer(
        model=model, args=args,
        train_dataset=train_ds, eval_dataset=test_ds,
        compute_metrics=compute_metrics,
    )
    trainer.train()
    trainer.evaluate()

    stress_test(model, tokenizer)


if __name__ == "__main__":
    main()
