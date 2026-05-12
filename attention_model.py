"""
attention_model.py
==================
Self-contained module containing the exact model and utilities from
softmax_kong_may07.ipynb

Contents:
  - AttentionPoolingModel  : softmax attention pooling (unchanged from notebook)
  - PatientDataset         : PyTorch Dataset for per-patient .npy files
  - collate_fn             : variable-length padding collator
  - load_patient_folder    : load a folder of .npy embeddings + labels array
  - load_all_patients_flat : load all donors for 5-fold CV splitting
  - set_seed               : reproducibility helper

Usage
-----
    from attention_model import (
        AttentionPoolingModel,
        PatientDataset,
        collate_fn,
        load_patient_folder,
        load_all_patients_flat,
        set_seed,
    )
"""

import os
import random

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset


# ─────────────────────────────────────────────────────────────────
#  Reproducibility
# ─────────────────────────────────────────────────────────────────

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark     = False
    os.environ["PYTHONHASHSEED"]       = str(seed)
    print(f"Random seed set to {seed} ✓")


# ─────────────────────────────────────────────────────────────────
#  Model — exact copy from notebook Block 3
# ─────────────────────────────────────────────────────────────────

class AttentionPoolingModel(nn.Module):
    """
    Aggregates a variable number of cell embeddings into one
    patient-level embedding via learned softmax attention,
    then classifies the patient.

    Args:
        d_h         : Dimensionality of input cell embeddings
        num_classes : Number of output classes
    """

    def __init__(self, d_h: int, num_classes: int):
        super().__init__()

        # ── Attention network ──────────────────────────────────────────
        # Input : (B, N, d_h)
        # Output: (B, N, 1) → scalar importance score per cell
        self.attn = nn.Sequential(
            nn.Linear(d_h, 256),  # Mix features across the latent space
            nn.Tanh(),            # Bounded non-linearity: output in (-1, +1)
            nn.Linear(256, 1)     # Collapse to a single score per cell
        )

        # ── Classifier ────────────────────────────────────────────────
        # Input : (B, d_h)         — pooled patient embedding
        # Output: (B, num_classes) — raw logits for CrossEntropyLoss
        self.classifier = nn.Sequential(
            nn.Linear(d_h, 256),
            nn.ReLU(),
            nn.Dropout(0.0),
            nn.Linear(256, num_classes)
        )

    def forward(self, Z, mask=None):
        """
        Args:
            Z    : (B, N, d_h) — batch of padded cell embedding matrices
            mask : (B, N) bool  — True = real cell, False = padding

        Returns:
            preds       : (B, num_classes) — classification logits
            patient_emb : (B, d_h)         — pooled patient embedding
            weights     : (B, N)           — per-cell attention weights (sum to 1)
        """

        # Step 1: Score every cell
        # attn maps (B, N, d_h) → (B, N, 1); squeeze removes last dim → (B, N)
        scores = self.attn(Z).squeeze(-1)

        # Step 2: Zero out padded positions
        # Setting padding scores to -1e9 makes softmax assign ~0 weight there
        if mask is not None:
            scores = scores.masked_fill(~mask, -1e9)

        # Step 3: Convert raw scores → probability weights
        # softmax over cell dimension → weights sum to 1 per patient
        weights = F.softmax(scores, dim=1)  # (B, N)

        # Step 4: Weighted sum → single patient embedding
        # weights.unsqueeze(-1) → (B, N, 1); broadcast-multiply with Z (B, N, d_h)
        patient_emb = torch.sum(
            Z * weights.unsqueeze(-1), dim=1
        )  # → (B, d_h)

        # Step 5: Classify the patient
        preds = self.classifier(patient_emb)  # (B, num_classes)

        return preds, patient_emb, weights


# ─────────────────────────────────────────────────────────────────
#  Dataset — exact copy from notebook Block 4
# ─────────────────────────────────────────────────────────────────

class PatientDataset(Dataset):
    """
    Wraps a list of patient dicts into a PyTorch Dataset.

    Each dict must contain:
        'Z'     : FloatTensor of shape (num_cells, d_h)
        'mask'  : BoolTensor  of shape (num_cells,)
        'label' : int
    """

    def __init__(self, data_list):
        self.data = data_list

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        return (
            item["Z"],
            item["mask"],
            item["label"],
            item["donor_id"]
        )


# ─────────────────────────────────────────────────────────────────
#  Collate — exact copy from notebook Block 5
# ─────────────────────────────────────────────────────────────────

def collate_fn(batch):
    """
    Pads a batch of variable-length patients to a uniform tensor.

    Args:
        batch : list of (Z, mask, label) tuples from PatientDataset

    Returns:
        Z_padded    : (B, max_cells, d_h)  — zero-padded cell embeddings
        mask_padded : (B, max_cells) bool  — True where cells are real
        labels      : (B,)                 — class labels
    """
    Z_list, mask_list, labels, donor_ids = zip(*batch)

    max_cells = max(z.shape[0] for z in Z_list)
    d_h = Z_list[0].shape[1]
    B   = len(Z_list)

    Z_padded    = torch.zeros(B, max_cells, d_h)
    mask_padded = torch.zeros(B, max_cells, dtype=torch.bool)

    for i, (Z, mask) in enumerate(zip(Z_list, mask_list)):
        n = Z.shape[0]
        Z_padded[i, :n]    = Z
        mask_padded[i, :n] = mask

    labels = torch.tensor(labels)

    return (
        Z_padded,
        mask_padded,
        labels,
        donor_ids
    )


# ─────────────────────────────────────────────────────────────────
#  Data loading helpers
# ─────────────────────────────────────────────────────────────────

def load_patient_folder(folder_path, labels_path):
    """
    Loads per-patient .npy files and labels.
    Each .npy: (num_cells, d_h)
    Returns list of dicts: Z, mask, label
    """
    data_list     = []
    labels        = np.load(labels_path)
    patient_files = sorted([
        f for f in os.listdir(folder_path)
        if f.endswith(".npy")
    ])

    assert len(patient_files) == len(labels), \
        f"Mismatch: {len(patient_files)} files, {len(labels)} labels"

    for i, fname in enumerate(patient_files):
        Z     = torch.tensor(
                    np.load(os.path.join(folder_path, fname)),
                    dtype=torch.float32
                )
        mask  = torch.ones(Z.shape[0], dtype=torch.bool)
        label = int(labels[i])
        data_list.append({"Z": Z, "mask": mask, "label": label})

    return data_list


def load_all_patients_flat(folder_path, labels_path):
    """
    Load all patients and return parallel lists for 5-fold CV splitting.

    Returns
    -------
    patient_list  : list of dicts (Z, mask, label)  — length N
    donor_ids     : list of str donor IDs            — length N
    labels_array  : np.ndarray int64                 — length N
    """
    labels        = np.load(labels_path)
    patient_files = sorted([
        f for f in os.listdir(folder_path)
        if f.endswith(".npy")
    ])

    assert len(patient_files) == len(labels), \
        f"Mismatch: {len(patient_files)} files, {len(labels)} labels"

    patient_list = []
    donor_ids    = []

    for i, fname in enumerate(patient_files):
        Z     = torch.tensor(
                    np.load(os.path.join(folder_path, fname)),
                    dtype=torch.float32
                )
        mask  = torch.ones(Z.shape[0], dtype=torch.bool)
        patient_list.append({
            "Z": Z,
            "mask": mask,
            "label": int(labels[i]),
            "donor_id": fname.replace("_embeddings.npy", "")
        })
        donor_ids.append(fname.replace("_embeddings.npy", ""))

    return patient_list, donor_ids, labels
