"""
Steps 1 + 2 of the DE/fgsea pipeline — SAP (HA-mirrored) variant.

Unlike HA, SAP has a SINGLE softmax attention pooling cells -> patient
(no cell-type hierarchy). Its raw `w_c` therefore sums to 1 over ALL
cells of a donor, not within each (donor, cell_type) group.

Step 1: For each of the 20 seeds, load the saved SAP checkpoint, run the
        outer-test donors through the model, and record the raw per-cell
        attention weight `w_c` (donor-level softmax) for every test cell.
        Aggregate to one row per cell with mean(w_c) over the seeds in
        which the cell was an outer-test cell.

Step 2: Within each (donor_id, cell_type) group, RE-NORMALISE the
        mean `w_c` so it sums to 1 within the group (this mirrors what
        HA's Level-1 attention does by construction). Then sort cells
        by w_c_mean descending and keep the smallest set whose cumulative
        (renormalised) mass is >= 0.80. These are the "high-attention"
        cells that feed the per-celltype DE step.

Outputs (under de_fgsea_results/):
  - wc_per_cell_per_seed.parquet   long: (obs_idx, donor_id, cell_type, seed, w_c)
  - wc_per_cell_mean.parquet       per cell: (obs_idx, donor_id, cell_type,
                                              disease_label, w_c_mean, n_seeds_seen)
  - high_attn_cells.parquet        per cell: same + cum_mass + group_size
  - high_attn_summary.csv          per (cell_type, label) counts
"""

import os, warnings, json, time
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.model_selection import StratifiedShuffleSplit
import scanpy as sc
from torch_geometric.utils import softmax as pyg_softmax
from torch_geometric.nn import global_add_pool, global_mean_pool

# ── Config ─────────────────────────────────────────────────────────────
HERE = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(HERE, "..", "data_healthy_inflamed",
                         "kong_healthy_inflamed_full_genes.h5ad")
CKPT_DIR  = os.path.join(HERE, "checkpoints_20seeds")
OUT_DIR   = os.path.join(HERE, "de_fgsea_results")
os.makedirs(OUT_DIR, exist_ok=True)

PATIENT_ID_KEY = "donor_id"
LABEL_KEY      = "Type"
CELL_TYPE_KEY  = "Celltype"
EMBEDDING_KEY  = "X_scGPT"
LABEL_MAP      = {"Heal": 0, "Infl": 1}
LABEL_NAMES    = {0: "Normal", 1: "Inflamed"}
NUM_CLASSES    = 2

ATTN = True
N_HID = 256; N_LAYERS_LIN = 1; DROPOUT = 0.3
TEST_SIZE = 6
N_SEEDS   = 20

CUM_MASS  = 0.80
DEVICE    = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("device:", DEVICE)


class SAPModel(nn.Module):
    """
    Softmax Attention Pooling MIL model (HA-mirrored: single Linear head
    so per-cell logit decomposition is exact). Single-level attention:
    one softmax over all cells of a patient -> patient embedding.

    Must match the architecture used to train the checkpoints in
    SAP_healthy_inflamed/checkpoints_20seeds/.
    """
    def __init__(self, n_in, n_out=2, attn=True,
                 dropout=0.0, n_layers_lin=1, n_hid=256):
        super().__init__()
        self.attn = attn
        layers = []
        for i in range(n_layers_lin):
            c_in = n_in if i == 0 else n_hid
            layers += [nn.Linear(c_in, n_hid), nn.ReLU(), nn.Dropout(dropout)]
        self.lin = nn.Sequential(*layers)
        curr_in = n_in if len(self.lin) == 0 else n_hid
        self.n_in1 = curr_in
        self.w_c   = nn.Sequential(nn.Linear(curr_in, 1), nn.Dropout(dropout))
        self.lin_out = nn.Linear(curr_in, n_out)

    def forward(self, X, batch, n_patients):
        """
        X       : (n_cells_total, n_in) cells stacked across patients
        batch   : (n_cells_total,) long  patient index per cell
        Returns : (logits[n_patients, n_out], w_c[n_cells_total])
        """
        X = self.lin(X)
        if self.attn:
            w_c_scores = self.w_c(X).squeeze(-1)
            w_c        = pyg_softmax(w_c_scores, batch)
            X_pat      = global_add_pool(X * w_c.unsqueeze(-1), batch, size=n_patients)
        else:
            w_c, X_pat = None, global_mean_pool(X, batch, size=n_patients)
        return self.lin_out(X_pat), w_c


# ── Load data ──────────────────────────────────────────────────────────
print("loading adata ...")
t0 = time.time()
adata = sc.read_h5ad(DATA_PATH)
print(f"  loaded in {time.time()-t0:.1f}s")
adata.obs["label"] = adata.obs[LABEL_KEY].map(LABEL_MAP).astype(int)

ALL_CT  = sorted(adata.obs[CELL_TYPE_KEY].unique().tolist())
N_CT    = len(ALL_CT)
CT_DICT = {ct: i for i, ct in enumerate(ALL_CT)}
embeddings = adata.obsm[EMBEDDING_KEY]
N_FEATURES = embeddings.shape[1]

df = pd.DataFrame(embeddings, index=adata.obs.index)
df["patient"]              = adata.obs[PATIENT_ID_KEY].values
df["cell_type_annotation"] = adata.obs[CELL_TYPE_KEY].values
df["label"]                = adata.obs["label"].values
df["__obs_idx__"]          = np.arange(len(df))

samples    = df[["patient", "label"]].drop_duplicates().reset_index(drop=True)
all_labels = samples["label"].values
print(f"{len(samples)} donors, {N_CT} cell types, embedding dim {N_FEATURES}, "
      f"total cells {len(df)}")


def get_data_with_index(df, samples_sub):
    """SAP batch encoding: one ID per patient (no cell-type sub-grouping)."""
    Xs, batches, pt_idx, ct_idx, obs_idx = [], [], [], [], []
    for idx, patient in enumerate(samples_sub["patient"].tolist()):
        sub = df[df["patient"] == patient]
        x   = sub.iloc[:, :N_FEATURES].to_numpy()
        cts = [CT_DICT[c] for c in sub["cell_type_annotation"].tolist()]
        Xs.append(x)
        batches.append(np.full(len(sub), idx, dtype=np.int64))
        pt_idx.append(np.full(len(sub), idx, dtype=np.int64))
        ct_idx.append(np.array(cts, dtype=np.int64))
        obs_idx.append(sub["__obs_idx__"].to_numpy())
    return (torch.tensor(np.concatenate(Xs), dtype=torch.float),
            torch.tensor(np.concatenate(batches), dtype=torch.long),
            np.concatenate(pt_idx), np.concatenate(ct_idx),
            np.concatenate(obs_idx))


# ── Step 1: extract w_c per cell per seed ─────────────────────────────
@torch.no_grad()
def extract_seed_wc(seed_id):
    ckpt = os.path.join(CKPT_DIR,
                        f"best_sap_ha_mirrored_model_seed{seed_id:02d}.pt")
    sss  = StratifiedShuffleSplit(n_splits=1, test_size=TEST_SIZE, random_state=seed_id)
    _, ote_idx = next(sss.split(np.arange(len(samples)), all_labels))
    ote_samples = samples.iloc[ote_idx].reset_index(drop=True)

    X, b, pt_idx, ct_idx, obs_idx = get_data_with_index(df, ote_samples)
    X, b = X.to(DEVICE), b.to(DEVICE)
    n_patients = len(ote_samples)

    model = SAPModel(n_in=N_FEATURES, n_out=NUM_CLASSES, attn=ATTN,
                     dropout=DROPOUT, n_layers_lin=N_LAYERS_LIN, n_hid=N_HID).to(DEVICE)
    model.load_state_dict(torch.load(ckpt, map_location=DEVICE, weights_only=True))
    model.eval()

    _, w_c = model(X, b, n_patients)
    w_c_np = w_c.cpu().numpy()

    out = pd.DataFrame({
        "obs_idx":   obs_idx,
        "donor_id":  ote_samples["patient"].values[pt_idx].astype(str),
        "cell_type": [ALL_CT[c] for c in ct_idx],
        "label":     ote_samples["label"].values[pt_idx],
        "seed":      seed_id,
        "w_c":       w_c_np,
    })
    return out


per_seed_path = os.path.join(OUT_DIR, "wc_per_cell_per_seed.parquet")
if os.path.exists(per_seed_path):
    print(f"\n[step1] using cached {per_seed_path}")
    long = pd.read_parquet(per_seed_path)
else:
    print("\n[step1] extracting w_c for all 20 seeds ...")
    rows = []
    for s in range(N_SEEDS):
        t0 = time.time()
        sub = extract_seed_wc(s)
        rows.append(sub)
        print(f"  seed {s:>2}: {len(sub):>6} cells  "
              f"donors={sub['donor_id'].nunique()}  "
              f"elapsed={time.time()-t0:.1f}s")
    long = pd.concat(rows, ignore_index=True)
    long.to_parquet(per_seed_path, index=False)
    print(f"  saved -> {per_seed_path}  rows={len(long):,}")

# ── Step 1b: per-cell mean(w_c) across seeds ──────────────────────────
print("\n[step1b] aggregating to per-cell mean w_c ...")
mean_df = (long.groupby(["obs_idx", "donor_id", "cell_type", "label"], observed=True)
                .agg(w_c_mean=("w_c", "mean"),
                     n_seeds_seen=("seed", "nunique"))
                .reset_index())
mean_df["disease_label"] = mean_df["label"].map(LABEL_NAMES)
mean_path = os.path.join(OUT_DIR, "wc_per_cell_mean.parquet")
mean_df.to_parquet(mean_path, index=False)
print(f"  cells with attention coverage: {len(mean_df):,}  "
      f"donors={mean_df['donor_id'].nunique()}  "
      f"cell_types={mean_df['cell_type'].nunique()}")
print(f"  saved -> {mean_path}")

print("\n  seed-coverage per cell (count):")
print(mean_df["n_seeds_seen"].describe().to_string())

# ── Step 2: top-80% cumulative w_c mass per (donor, cell_type) ────────
# NOTE for SAP: raw w_c is a donor-level softmax (sums to 1 over ALL of a
# donor's cells). pick_high_attn re-normalises within each (donor, cell_type)
# group below, mirroring what HA's Level-1 attention does by construction.
print(f"\n[step2] selecting top {int(CUM_MASS*100)}% cumulative w_c mass per "
      f"(donor, cell_type)  [renormalised within group] ...")


def pick_high_attn(group):
    g = group.sort_values("w_c_mean", ascending=False).copy()
    total = g["w_c_mean"].sum()
    if total <= 0:
        return g.iloc[0:0]
    g["norm_w"] = g["w_c_mean"] / total
    g["cum_mass"] = g["norm_w"].cumsum()
    # keep smallest prefix whose cum_mass >= CUM_MASS
    keep_n = int((g["cum_mass"] >= CUM_MASS).idxmax()
                 - g.index[0] + 1) if (g["cum_mass"] >= CUM_MASS).any() else len(g)
    return g.iloc[:keep_n]


high = (mean_df.groupby(["donor_id", "cell_type"], group_keys=False, observed=True)
                .apply(pick_high_attn))
high["group_size"] = high.groupby(["donor_id", "cell_type"], observed=True)["obs_idx"].transform("size")
high_path = os.path.join(OUT_DIR, "high_attn_cells.parquet")
high.to_parquet(high_path, index=False)
print(f"  kept {len(high):,} / {len(mean_df):,} cells "
      f"({len(high)/len(mean_df)*100:.1f}%)")
print(f"  saved -> {high_path}")

# Per-(cell_type, label) summary
summary = (high.groupby(["cell_type", "disease_label"], observed=True)
                .agg(n_cells=("obs_idx", "size"),
                     n_donors=("donor_id", "nunique"))
                .reset_index()
                .pivot_table(index="cell_type",
                             columns="disease_label",
                             values=["n_cells", "n_donors"],
                             fill_value=0))
summary.columns = [f"{a}_{b}" for a, b in summary.columns]
summary = summary.reset_index().sort_values("cell_type")
summary_path = os.path.join(OUT_DIR, "high_attn_summary.csv")
summary.to_csv(summary_path, index=False)
print(f"\n[step2] summary saved -> {summary_path}")
print(summary.head(15).to_string(index=False))

# ── Per-cell-type overall summary (totals, not split by disease) ──────────
# Denominators:
#   total_cells_in_data : every cell of that CT across the FULL adata
#   total_cells_scored  : cells of that CT that were ever an outer-test cell
#                         (= the denominator that high-attention selection saw)
total_in_data   = df.groupby("cell_type_annotation").size().rename("total_cells_in_data")
donors_in_data  = df.groupby("cell_type_annotation")["patient"].nunique().rename("num_donors_in_data")
total_scored    = mean_df.groupby("cell_type", observed=True).size().rename("total_cells_scored")
high_per_ct     = high.groupby("cell_type", observed=True).agg(
    cells_with_high_attention=("obs_idx", "size"),
    num_donors_high_attn=("donor_id", "nunique"),
)
ct_overview = (high_per_ct
               .join(total_scored, how="outer")
               .join(total_in_data, how="outer")
               .join(donors_in_data, how="outer")
               .fillna(0)
               .astype({"total_cells_in_data": int, "num_donors_in_data": int,
                        "total_cells_scored": int, "cells_with_high_attention": int,
                        "num_donors_high_attn": int}))
ct_overview["pct_high_of_scored"] = np.where(
    ct_overview["total_cells_scored"] > 0,
    100.0 * ct_overview["cells_with_high_attention"] / ct_overview["total_cells_scored"],
    0.0,
).round(2)
ct_overview = (ct_overview
               .reset_index()
               .rename(columns={"index": "cell_type"})
               [["cell_type", "total_cells_in_data", "total_cells_scored",
                 "cells_with_high_attention", "pct_high_of_scored",
                 "num_donors_in_data", "num_donors_high_attn"]]
               .sort_values("cells_with_high_attention", ascending=False))
overview_path = os.path.join(OUT_DIR, "high_attn_celltype_overview.csv")
ct_overview.to_csv(overview_path, index=False)
print(f"\n[step2] per-cell-type overview saved -> {overview_path}")
print(ct_overview.head(15).to_string(index=False))

print("\nDone.")
