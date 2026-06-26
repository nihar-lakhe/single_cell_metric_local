"""
Wilcoxon rank-sum DE per cell type — Inflamed vs Normal.

Two modes (--cells):
  high_attn : restrict to cells listed in
              ../de_fgsea_results/high_attn_cells.parquet (SAP top-80% mass)
  all       : every cell in the full-gene AnnData

Outputs (results/<mode>/):
  de_wilcox/<cell_type>.csv        per-gene: gene, logFC, AUC, pct_normal,
                                   pct_inflamed, U_stat, p_value, padj_BH
  de_wilcox_combined.csv           concatenated
  celltype_eligibility.csv         per-CT counts + eligible flag

Eligibility (same as MAST pipeline):
  >= 10 cells per disease group AND >= 3 donors per disease group.

Normalisation:
  log1p(CP10k) — same as export_high_attn_for_seurat.py / Seurat default.
"""

import os, sys, argparse, time, warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import scanpy as sc
import scipy.sparse as sp
from scipy.stats import mannwhitneyu
from statsmodels.stats.multitest import multipletests

# ── Paths ──────────────────────────────────────────────────────────────────
HERE      = os.path.dirname(os.path.abspath(__file__))
SAP_ROOT  = os.path.abspath(os.path.join(HERE, ".."))
DATA_PATH = os.path.join(SAP_ROOT, "..", "data_healthy_inflamed",
                         "kong_healthy_inflamed_full_genes.h5ad")
HIGH_ATTN_PATH = os.path.join(SAP_ROOT, "de_fgsea_results", "high_attn_cells.parquet")

PATIENT_ID_KEY = "donor_id"
LABEL_KEY      = "Type"
CELL_TYPE_KEY  = "Celltype"
LABEL_MAP      = {"Heal": 0, "Infl": 1}
LABEL_NAMES    = {0: "Normal", 1: "Inflamed"}

# Eligibility (matched to de_mast_fgsea.R)
MIN_CELLS_PER_GROUP   = 10
MIN_DONORS_PER_GROUP  = 3

# Optional CT cap (set very high to effectively disable; full set still tractable
# in pure Wilcoxon).
MAX_CELLS_PER_CT = 200_000


def cp10k_log1p(X):
    """log1p(CP10k) on a sparse matrix (cells x genes), return CSR float32."""
    X = X.tocsr().astype(np.float32)
    counts = np.asarray(X.sum(axis=1)).ravel()
    counts[counts == 0] = 1.0
    inv = (1e4 / counts).astype(np.float32)
    # row-scale by inv via diagonal multiply
    D = sp.diags(inv)
    X = D @ X
    X.data = np.log1p(X.data)
    return X.tocsr()


def wilcox_one_celltype(X_norm, group_labels, gene_names, ct):
    """
    X_norm        : (n_cells, n_genes) CSR float32, log1p(CP10k)
    group_labels  : (n_cells,) 0=Normal, 1=Inflamed
    Returns a DataFrame with one row per gene that was tested.
    """
    is_norm = (group_labels == 0)
    is_infl = (group_labels == 1)
    n_norm = int(is_norm.sum()); n_infl = int(is_infl.sum())

    # Drop genes expressed in 0 cells of this subset (no test possible)
    nz_mask = np.asarray((X_norm > 0).sum(axis=0)).ravel() > 0
    if not nz_mask.any():
        return pd.DataFrame(columns=["gene", "logFC", "AUC",
                                     "pct_normal", "pct_inflamed",
                                     "U_stat", "p_value", "padj_BH",
                                     "cell_type"])
    X_norm = X_norm[:, nz_mask]
    genes  = np.asarray(gene_names)[nz_mask]

    # Densify per gene to compute mannwhitneyu — operate column by column to
    # keep peak memory at one column at a time
    n_genes = X_norm.shape[1]
    logFC       = np.zeros(n_genes, dtype=np.float32)
    AUC         = np.zeros(n_genes, dtype=np.float32)
    pct_norm    = np.zeros(n_genes, dtype=np.float32)
    pct_infl    = np.zeros(n_genes, dtype=np.float32)
    U_stat      = np.zeros(n_genes, dtype=np.float32)
    p_value     = np.ones (n_genes, dtype=np.float64)

    # Pre-extract indices for speed
    idx_norm = np.where(is_norm)[0]
    idx_infl = np.where(is_infl)[0]

    # Densify in batches to amortise overhead
    BATCH = 500
    t0 = time.time()
    for start in range(0, n_genes, BATCH):
        stop = min(start + BATCH, n_genes)
        block = X_norm[:, start:stop].toarray()  # (n_cells, batch)
        # mean log-expression per group
        m_norm = block[idx_norm].mean(axis=0)
        m_infl = block[idx_infl].mean(axis=0)
        logFC[start:stop] = m_infl - m_norm
        # pct expressing
        pct_norm[start:stop] = (block[idx_norm] > 0).mean(axis=0) * 100.0
        pct_infl[start:stop] = (block[idx_infl] > 0).mean(axis=0) * 100.0
        # Mann-Whitney U per gene
        for j in range(stop - start):
            x = block[idx_norm, j]
            y = block[idx_infl, j]
            # If both sides identical -> NaN p
            if x.min() == x.max() and y.min() == y.max() and x[0] == y[0]:
                U_stat[start + j] = 0.5 * n_norm * n_infl
                p_value[start + j] = 1.0
                AUC[start + j] = 0.5
                continue
            try:
                res = mannwhitneyu(y, x, alternative="two-sided",
                                   method="asymptotic")  # Inflamed vs Normal
                U_stat[start + j] = res.statistic
                p_value[start + j] = res.pvalue
                # AUC of Inflamed > Normal
                AUC[start + j] = res.statistic / (n_norm * n_infl)
            except Exception:
                U_stat[start + j] = np.nan
                p_value[start + j] = np.nan
                AUC[start + j] = np.nan

    valid = ~np.isnan(p_value)
    padj = np.full_like(p_value, np.nan)
    if valid.any():
        padj[valid] = multipletests(p_value[valid], method="fdr_bh")[1]

    out = pd.DataFrame({
        "gene":         genes,
        "logFC":        logFC,
        "AUC":          AUC,
        "pct_normal":   pct_norm,
        "pct_inflamed": pct_infl,
        "U_stat":       U_stat,
        "p_value":      p_value,
        "padj_BH":      padj,
        "cell_type":    ct,
    })
    out = out.sort_values("p_value").reset_index(drop=True)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cells", choices=["high_attn", "all"], required=True,
                    help="Which cell set to test on.")
    args = ap.parse_args()

    OUT_DIR = os.path.join(HERE, "results", "high_attn" if args.cells == "high_attn" else "all_cells")
    OUT_DE  = os.path.join(OUT_DIR, "de_wilcox")
    os.makedirs(OUT_DE, exist_ok=True)

    print(f"[mode] {args.cells}")
    print(f"[out ] {OUT_DIR}")

    # ── Load data ─────────────────────────────────────────────────────────
    print("loading adata ...")
    t0 = time.time()
    adata = sc.read_h5ad(DATA_PATH)
    print(f"  loaded in {time.time()-t0:.1f}s   ({adata.n_obs:,} cells x {adata.n_vars:,} genes)")
    adata.obs["label"] = adata.obs[LABEL_KEY].map(LABEL_MAP).astype(int)
    adata.obs["disease_label"] = adata.obs["label"].map(LABEL_NAMES)
    adata.obs["obs_idx"] = np.arange(adata.n_obs)

    # ── Subset to high-attention cells if requested ───────────────────────
    if args.cells == "high_attn":
        if not os.path.exists(HIGH_ATTN_PATH):
            sys.exit(f"ERROR: {HIGH_ATTN_PATH} not found. Run extract_wc_and_select_high_attn.py first.")
        ha = pd.read_parquet(HIGH_ATTN_PATH)[["obs_idx"]].drop_duplicates()
        keep_mask = np.zeros(adata.n_obs, dtype=bool)
        keep_mask[ha["obs_idx"].to_numpy()] = True
        adata = adata[keep_mask].copy()
        print(f"  subset to {adata.n_obs:,} high-attention cells")

    # ── Eligibility ───────────────────────────────────────────────────────
    elig_rows = []
    for ct, sub in adata.obs.groupby(CELL_TYPE_KEY):
        n_norm = int((sub["disease_label"] == "Normal").sum())
        n_infl = int((sub["disease_label"] == "Inflamed").sum())
        d_norm = int(sub.loc[sub["disease_label"] == "Normal", PATIENT_ID_KEY].nunique())
        d_infl = int(sub.loc[sub["disease_label"] == "Inflamed", PATIENT_ID_KEY].nunique())
        eligible = (n_norm >= MIN_CELLS_PER_GROUP and n_infl >= MIN_CELLS_PER_GROUP and
                    d_norm >= MIN_DONORS_PER_GROUP and d_infl >= MIN_DONORS_PER_GROUP)
        elig_rows.append({"cell_type": ct,
                          "n_cells_Normal": n_norm, "n_cells_Inflamed": n_infl,
                          "n_donors_Normal": d_norm, "n_donors_Inflamed": d_infl,
                          "eligible": eligible})
    elig = pd.DataFrame(elig_rows).sort_values("cell_type")
    elig.to_csv(os.path.join(OUT_DIR, "celltype_eligibility.csv"), index=False)
    cts = elig.loc[elig["eligible"], "cell_type"].tolist()
    print(f"eligible cell types: {len(cts)} / {len(elig)} "
          f"(>={MIN_CELLS_PER_GROUP} cells & >={MIN_DONORS_PER_GROUP} donors per group)")

    # Sanity: cap big CTs (Wilcoxon is fast but memory grows linearly)
    safe_filename = lambda x: "".join(c if c.isalnum() or c in "._-" else "_" for c in x)

    # ── Normalise once on whole adata (CP10k log1p) ───────────────────────
    print("normalising counts (CP10k log1p) ...")
    t0 = time.time()
    if sp.issparse(adata.X):
        X_norm_full = cp10k_log1p(adata.X)
    else:
        X_norm_full = cp10k_log1p(sp.csr_matrix(adata.X))
    print(f"  normalised in {time.time()-t0:.1f}s")

    gene_names = adata.var_names.to_numpy()

    # ── Per-CT Wilcoxon ───────────────────────────────────────────────────
    all_de = []
    for i, ct in enumerate(cts, 1):
        t0 = time.time()
        cell_mask = (adata.obs[CELL_TYPE_KEY] == ct).to_numpy()
        if cell_mask.sum() > MAX_CELLS_PER_CT:
            sel = np.where(cell_mask)[0]
            sel = np.random.RandomState(42).choice(sel, MAX_CELLS_PER_CT, replace=False)
            cell_mask = np.zeros_like(cell_mask); cell_mask[sel] = True
            print(f"[{i:3}/{len(cts)} {ct}] capped to {MAX_CELLS_PER_CT} cells")

        X_ct  = X_norm_full[cell_mask]
        lbl   = adata.obs.loc[cell_mask, "label"].to_numpy()
        out   = wilcox_one_celltype(X_ct, lbl, gene_names, ct)
        if len(out) == 0:
            print(f"[{i:3}/{len(cts)} {ct}] no testable genes; skipping")
            continue
        out.to_csv(os.path.join(OUT_DE, safe_filename(ct) + ".csv"), index=False)
        n_sig = int((out["padj_BH"] < 0.05).sum())
        all_de.append(out)
        print(f"[{i:3}/{len(cts)} {ct}] {len(out):>5} genes  {n_sig:>5} FDR<0.05  "
              f"({time.time()-t0:.1f}s)")

    if all_de:
        combo = pd.concat(all_de, ignore_index=True)
        combo.to_csv(os.path.join(OUT_DIR, "de_wilcox_combined.csv"), index=False)
        print(f"\ncombined Wilcoxon DE rows: {len(combo):,}  "
              f"-> {os.path.join(OUT_DIR, 'de_wilcox_combined.csv')}")

    print("\nDone.")


if __name__ == "__main__":
    main()
