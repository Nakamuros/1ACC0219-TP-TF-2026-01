"""Fine-tune MentalRoBERTa for 3-class mental health classification.

Improvements over v1:
- Full fine-tuning (backbone unfrozen) with discriminative learning rates
- Correct LR range for transformers (2e-5 default)
- Linear warmup + cosine decay scheduler
- Class weights for imbalanced Suicidal class
- Gradient clipping
- Per-epoch validation metrics + best-model checkpoint
- Early stopping
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from torch.utils.data import DataLoader, Dataset
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    get_linear_schedule_with_warmup,
)


LABELS = {"Normal": 0, "Depression": 1, "Suicidal": 2}
ID2LABEL = {v: k for k, v in LABELS.items()}


class TextDataset(Dataset):
    def __init__(self, texts, labels):
        self.texts = list(texts)
        self.labels = list(labels)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        return self.texts[index], self.labels[index]


def build_collate(tokenizer, max_length):
    def collate(batch):
        batch_x, batch_y = zip(*batch)
        encoded = tokenizer(
            list(batch_x),
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )
        encoded["labels"] = torch.tensor(batch_y, dtype=torch.long)
        return encoded
    return collate


def evaluate(model, loader, device):
    model.eval()
    all_preds, all_labels = [], []
    with torch.inference_mode():
        for batch in loader:
            labels = batch.pop("labels").to(device)
            batch = {k: v.to(device) for k, v in batch.items()}
            logits = model(**batch).logits
            preds = logits.argmax(dim=1)
            all_preds.extend(preds.cpu().tolist())
            all_labels.extend(labels.cpu().tolist())
    return {
        "accuracy": accuracy_score(all_labels, all_preds),
        "macro_f1": f1_score(all_labels, all_preds, average="macro"),
        "report": classification_report(
            all_labels, all_preds, target_names=list(LABELS.keys())
        ),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="mental/mental-roberta-base")
    parser.add_argument("--output", default="modelo_mental_roberta")
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--warmup-ratio", type=float, default=0.1)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--patience", type=int, default=2,
                        help="Early stopping patience (epochs)")
    parser.add_argument("--max-train-samples", type=int, default=None)
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--freeze-layers", type=int, default=0,
                        help="Freeze bottom N encoder layers (0 = full fine-tuning)")
    args = parser.parse_args()

    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"Device: {device}")
    torch.manual_seed(42)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(42)

    # ── Data ──────────────────────────────────────────────────────────────────
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
            train_x, train_y,
            train_size=args.max_train_samples,
            stratify=train_y,
            random_state=42,
        )

    print(f"Train: {len(train_y)} | Val: {len(val_y)} | Test: {len(test_y)}")

    # ── Class weights (handles Suicidal imbalance) ────────────────────────────
    class_weights = compute_class_weight(
        class_weight="balanced", classes=np.unique(train_y), y=train_y
    )
    # Boost Suicidal (class 2) — same clinical rationale as LightGBM
    class_weights[2] *= 1.5
    weight_tensor = torch.tensor(class_weights, dtype=torch.float).to(device)
    print(f"Class weights: {dict(zip(ID2LABEL.values(), class_weights.round(3)))}")

    # ── Model & tokenizer ─────────────────────────────────────────────────────
    tokenizer = AutoTokenizer.from_pretrained(
        args.model, local_files_only=args.local_files_only
    )
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model,
        num_labels=3,
        id2label=ID2LABEL,
        label2id=LABELS,
        local_files_only=args.local_files_only,
    ).to(device)

    # Optional: freeze bottom N transformer layers for CPU training
    if args.freeze_layers > 0:
        for i, layer in enumerate(model.roberta.encoder.layer):
            if i < args.freeze_layers:
                for param in layer.parameters():
                    param.requires_grad = False
        print(f"Frozen layers: 0–{args.freeze_layers - 1} (fine-tuning {12 - args.freeze_layers} layers)")
    else:
        print("Full fine-tuning — all layers trainable")

    # ── Dataloaders ───────────────────────────────────────────────────────────
    collate = build_collate(tokenizer, args.max_length)
    train_loader = DataLoader(
        TextDataset(train_x, train_y),
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate,
    )
    val_loader = DataLoader(
        TextDataset(val_x, val_y),
        batch_size=args.batch_size * 2,
        shuffle=False,
        collate_fn=collate,
    )
    test_loader = DataLoader(
        TextDataset(test_x, test_y),
        batch_size=args.batch_size * 2,
        shuffle=False,
        collate_fn=collate,
    )

    # ── Optimizer & scheduler ─────────────────────────────────────────────────
    # Discriminative LR: lower LR for backbone, higher for classifier head
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_groups = [
        {
            "params": [
                p for n, p in model.roberta.named_parameters()
                if not any(nd in n for nd in no_decay) and p.requires_grad
            ],
            "lr": args.learning_rate,
            "weight_decay": 0.01,
        },
        {
            "params": [
                p for n, p in model.roberta.named_parameters()
                if any(nd in n for nd in no_decay) and p.requires_grad
            ],
            "lr": args.learning_rate,
            "weight_decay": 0.0,
        },
        {
            "params": model.classifier.parameters(),
            "lr": args.learning_rate * 5,  # classifier head trains faster
            "weight_decay": 0.01,
        },
    ]
    optimizer = torch.optim.AdamW(optimizer_groups)

    total_steps = len(train_loader) * args.epochs
    warmup_steps = int(total_steps * args.warmup_ratio)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps
    )

    loss_fn = torch.nn.CrossEntropyLoss(weight=weight_tensor)

    # ── Training loop ─────────────────────────────────────────────────────────
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    best_f1 = 0.0
    patience_counter = 0
    history = []

    for epoch in range(args.epochs):
        model.train()
        total_loss = 0.0
        for step, batch in enumerate(train_loader, start=1):
            labels_batch = batch.pop("labels").to(device)
            batch = {k: v.to(device) for k, v in batch.items()}

            optimizer.zero_grad(set_to_none=True)
            logits = model(**batch).logits
            loss = loss_fn(logits, labels_batch)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            optimizer.step()
            scheduler.step()

            total_loss += loss.item()
            if step % 100 == 0:
                print(
                    f"epoch={epoch+1}/{args.epochs} step={step}/{len(train_loader)} "
                    f"loss={total_loss/step:.4f} lr={scheduler.get_last_lr()[0]:.2e}",
                    flush=True,
                )

        val_metrics = evaluate(model, val_loader, device)
        print(
            f"\n── Epoch {epoch+1} val ── acc={val_metrics['accuracy']:.4f} "
            f"macro_f1={val_metrics['macro_f1']:.4f}\n{val_metrics['report']}",
            flush=True,
        )
        history.append({"epoch": epoch + 1, **{k: v for k, v in val_metrics.items() if k != "report"}})

        # Save best checkpoint
        if val_metrics["macro_f1"] > best_f1:
            best_f1 = val_metrics["macro_f1"]
            patience_counter = 0
            model.save_pretrained(output / "best")
            tokenizer.save_pretrained(output / "best")
            print(f"  ✓ Best model saved (macro_f1={best_f1:.4f})")
        else:
            patience_counter += 1
            print(f"  No improvement ({patience_counter}/{args.patience})")
            if patience_counter >= args.patience:
                print("Early stopping triggered.")
                break

    # ── Final evaluation with best checkpoint ─────────────────────────────────
    best_model = AutoModelForSequenceClassification.from_pretrained(output / "best").to(device)
    test_metrics = evaluate(best_model, test_loader, device)
    print(f"\n{'='*60}\nTEST SET RESULTS\n{'='*60}")
    print(f"Accuracy : {test_metrics['accuracy']:.4f}")
    print(f"Macro F1 : {test_metrics['macro_f1']:.4f}")
    print(test_metrics["report"])

    results = {
        "args": vars(args),
        "device": str(device),
        "training_history": history,
        "test": {k: v for k, v in test_metrics.items() if k != "report"},
        "test_report": test_metrics["report"],
    }
    (output / "metrics.json").write_text(json.dumps(results, indent=2), encoding="utf-8")

    # Also save final model (last epoch) alongside best
    model.save_pretrained(output / "last")
    tokenizer.save_pretrained(output / "last")
    print(f"\nArtifacts saved to '{output}/' (best/ and last/ subfolders)")


if __name__ == "__main__":
    main()
