"""
train.py - Training pipeline for the Basic Transformer (Encoder-Decoder) model.
Foundations of Generative AI (ICT4403) - Assignment 3
Implements:
  1. Forward pass with teacher forcing and cross-entropy loss (ignoring PAD_IDX)
  2. Adam optimizer with gradient clipping
  3. Validation loss and epoch checkpointing
  4. Training history logging (loss, perplexity, epoch runtimes)
"""

import os
import time
import json
import math
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from dataset import TranslationDataset, collate_fn, PAD_IDX

def train_epoch(model, dataloader, optimizer, criterion, clip_grad=1.0, device="cpu"):
    """Runs a single training epoch with teacher forcing."""
    model.train()
    epoch_loss = 0.0

    for batch_idx, (src, tgt, _, _) in enumerate(dataloader):
        src = src.to(device)
        tgt = tgt.to(device)

        # tgt input to decoder excludes the last token (<eos> or pad)
        tgt_input = tgt[:, :-1]
        # target output to calculate loss excludes the first token (<sos>)
        tgt_output = tgt[:, 1:]

        optimizer.zero_grad()
        logits = model(src, tgt_input)

        # Reshape logits: [batch_size * seq_len, vocab_size]
        logits = logits.contiguous().view(-1, logits.shape[-1])
        targets = tgt_output.contiguous().view(-1)

        loss = criterion(logits, targets)
        loss.backward()

        nn.utils.clip_grad_norm_(model.parameters(), clip_grad)
        optimizer.step()

        epoch_loss += loss.item()

    return epoch_loss / len(dataloader)


def evaluate_epoch(model, dataloader, criterion, device="cpu"):
    """Evaluates validation loss on unseen validation split."""
    model.eval()
    epoch_loss = 0.0

    with torch.no_grad():
        for src, tgt, _, _ in dataloader:
            src = src.to(device)
            tgt = tgt.to(device)

            tgt_input = tgt[:, :-1]
            tgt_output = tgt[:, 1:]

            logits = model(src, tgt_input)
            logits = logits.contiguous().view(-1, logits.shape[-1])
            targets = tgt_output.contiguous().view(-1)

            loss = criterion(logits, targets)
            epoch_loss += loss.item()

    return epoch_loss / len(dataloader)


def train_basic_transformer(
    model,
    train_pairs,
    val_pairs,
    src_vocab,
    tgt_vocab,
    num_epochs=15,
    batch_size=32,
    lr=0.0005,
    save_path="saved_models/basic_transformer.pt",
    history_path="results/training_history.json",
    device=None
):
    """
    Complete training loop for the Basic Transformer.
    Saves best model weights and exports detailed JSON history.
    """
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    os.makedirs(os.path.dirname(history_path), exist_ok=True)

    train_dataset = TranslationDataset(train_pairs, src_vocab, tgt_vocab)
    val_dataset = TranslationDataset(val_pairs, src_vocab, tgt_vocab)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

    criterion = nn.CrossEntropyLoss(ignore_index=PAD_IDX)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, betas=(0.9, 0.98), eps=1e-9)

    best_val_loss = float("inf")
    history = {
        "epochs": [],
        "train_loss": [],
        "val_loss": [],
        "train_ppl": [],
        "val_ppl": [],
        "epoch_times_sec": []
    }

    start_total_time = time.time()
    print(f"\n[Training] Starting Basic Transformer Training on {str(device).upper()} for {num_epochs} Epochs...")

    for epoch in range(1, num_epochs + 1):
        t0 = time.time()
        train_loss = train_epoch(model, train_loader, optimizer, criterion, device=device)
        val_loss = evaluate_epoch(model, val_loader, criterion, device=device)
        elapsed = time.time() - t0

        train_ppl = math.exp(min(train_loss, 20))
        val_ppl = math.exp(min(val_loss, 20))

        history["epochs"].append(epoch)
        history["train_loss"].append(round(train_loss, 4))
        history["val_loss"].append(round(val_loss, 4))
        history["train_ppl"].append(round(train_ppl, 2))
        history["val_ppl"].append(round(val_ppl, 2))
        history["epoch_times_sec"].append(round(elapsed, 2))

        print(f"Epoch {epoch:02d}/{num_epochs:02d} | Train Loss: {train_loss:.4f} (PPL: {train_ppl:.2f}) | Val Loss: {val_loss:.4f} (PPL: {val_ppl:.2f}) | Time: {elapsed:.2f}s")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_loss": val_loss,
                "src_vocab_size": src_vocab.n_words,
                "tgt_vocab_size": tgt_vocab.n_words
            }, save_path)
            print(f"  --> Checkpoint saved to {save_path} (Best Val Loss: {val_loss:.4f})")

    total_training_time = round(time.time() - start_total_time, 2)
    history["total_training_time_sec"] = total_training_time
    history["best_val_loss"] = round(best_val_loss, 4)

    with open(history_path, "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)

    print(f"[Training Complete] Total time: {total_training_time}s | Best Val Loss: {best_val_loss:.4f}\n")
    return history
