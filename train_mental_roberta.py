"""Fine-tune the MentalRoBERTa classification head for the three project labels."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
import torch
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer


LABELS = {"Normal": 0, "Depression": 1, "Suicidal": 2}


class TextDataset(Dataset):
    def __init__(self, texts, labels):
        self.texts = list(texts)
        self.labels = list(labels)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        return self.texts[index], self.labels[index]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="mental/mental-roberta-base")
    parser.add_argument("--output", default="modelo_mental_roberta")
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--max-length", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--max-train-samples", type=int, default=None)
    parser.add_argument("--local-files-only", action="store_true")
    args = parser.parse_args()

    torch.manual_seed(42)
    torch.set_num_threads(min(16, torch.get_num_threads()))

    frame = pd.read_csv("mental_health_cleaned.csv", usecols=["statement", "status"])
    frame = frame[frame.status.isin(LABELS)].dropna(subset=["statement"])
    texts = frame.statement.astype(str).to_numpy()
    labels = frame.status.map(LABELS).to_numpy()
    train_x, temp_x, train_y, temp_y = train_test_split(
        texts, labels, test_size=0.30, stratify=labels, random_state=42
    )
    val_x, test_x, val_y, test_y = train_test_split(
        temp_x, temp_y, test_size=0.50, stratify=temp_y, random_state=42
    )
    if args.max_train_samples and args.max_train_samples < len(train_y):
        train_x, _, train_y, _ = train_test_split(
            train_x, train_y, train_size=args.max_train_samples,
            stratify=train_y, random_state=42
        )

    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=args.local_files_only)
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model,
        num_labels=3,
        id2label={value: key for key, value in LABELS.items()},
        label2id=LABELS,
        local_files_only=args.local_files_only,
    )
    # CPU-friendly transfer learning: retain MentalRoBERTa representations and
    # train the newly initialized classification head.
    for parameter in model.roberta.parameters():
        parameter.requires_grad = False

    def collate(batch):
        batch_x, batch_y = zip(*batch)
        encoded = tokenizer(
            list(batch_x), padding=True, truncation=True,
            max_length=args.max_length, return_tensors="pt"
        )
        encoded["labels"] = torch.tensor(batch_y, dtype=torch.long)
        return encoded

    train_loader = DataLoader(
        TextDataset(train_x, train_y), batch_size=args.batch_size,
        shuffle=True, collate_fn=collate
    )
    optimizer = torch.optim.AdamW(model.classifier.parameters(), lr=args.learning_rate)
    model.train()
    for epoch in range(args.epochs):
        total_loss = 0.0
        for step, batch in enumerate(train_loader, start=1):
            optimizer.zero_grad(set_to_none=True)
            output = model(**batch)
            output.loss.backward()
            optimizer.step()
            total_loss += output.loss.item()
            if step % 100 == 0:
                print(
                    f"epoch={epoch + 1}/{args.epochs} step={step}/{len(train_loader)} "
                    f"loss={total_loss / step:.4f}", flush=True
                )

    def evaluate(eval_x, eval_y):
        loader = DataLoader(
            TextDataset(eval_x, eval_y), batch_size=args.batch_size,
            shuffle=False, collate_fn=collate
        )
        predictions = []
        model.eval()
        with torch.inference_mode():
            for batch in loader:
                labels_batch = batch.pop("labels")
                predictions.extend(model(**batch).logits.argmax(dim=1).tolist())
        return {
            "accuracy": accuracy_score(eval_y, predictions),
            "macro_f1": f1_score(eval_y, predictions, average="macro"),
        }

    metrics = {"validation": evaluate(val_x, val_y), "test": evaluate(test_x, test_y)}
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(output)
    tokenizer.save_pretrained(output)
    (output / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(json.dumps(metrics, indent=2), flush=True)


if __name__ == "__main__":
    main()
