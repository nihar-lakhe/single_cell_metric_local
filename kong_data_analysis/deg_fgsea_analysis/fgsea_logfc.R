#!/usr/bin/env Rscript
# ---------------------------------------------------------------------------
# fgsea using MSigDB C7 (immunologic) and H (Hallmark) on Wilcoxon DE results.
#
# logFC-ranked variant of fgsea_c7.R: genes are ranked by their raw logFC
# (effect size), NOT by signed -log10(p). Reuses the existing Wilcoxon DE
# CSVs (no DEG re-run) and writes to a SEPARATE results tree so the original
# p-value-ranked results are left untouched.
#
# Usage:
#   Rscript fgsea_logfc.R --mode high_attn [--collections c7,h]
#   Rscript fgsea_logfc.R --mode all       [--collections c7,h]
#
# Inputs (reused, read-only):
#   results/<mode>/de_wilcox/<cell_type>.csv          per-CT Wilcoxon DE
#   ../de_fgsea_results/gmt/c7.all.v7.0.symbols.gmt   MSigDB C7
#   ../de_fgsea_results/gmt/h.all.v7.0.symbols.gmt    MSigDB Hallmark
#
# Outputs (results_logfc/<mode>/), per collection <coll> in {c7, h}:
#   fgsea_<coll>/<cell_type>.csv                      per-CT fgsea
#   fgsea_<coll>_combined.csv
#   fgsea_<coll>_signed_neglog10padj_matrix.csv       pathway x cell_type heatmap
#   fgsea_<coll>_pathway_summary.csv                  pathway frac_sig, n_up/dn
#
# Ranks for fgsea:
#   raw logFC   (Inflamed - Normal mean log1p-CP10k; positive = up in Inflamed)
# ---------------------------------------------------------------------------

options(expressions = 500000)
suppressPackageStartupMessages({
  library(data.table)
  library(fgsea)
  library(BiocParallel)
})

# ── CLI ────────────────────────────────────────────────────────────────────
args <- commandArgs(trailingOnly = TRUE)
mode <- NULL
colls_arg <- "c7,h"
for (i in seq_along(args)) {
  if (args[i] == "--mode" && i < length(args))         mode      <- args[i + 1]
  if (args[i] == "--collections" && i < length(args))  colls_arg <- args[i + 1]
}
if (is.null(mode) || !(mode %in% c("high_attn", "all"))) {
  stop("Usage: Rscript fgsea_logfc.R --mode {high_attn|all} [--collections c7,h]")
}
COLLECTIONS <- trimws(strsplit(colls_arg, ",")[[1]])
stopifnot(all(COLLECTIONS %in% c("c7", "h", "c7h")))

HERE     <- dirname(sub("--file=", "", grep("--file=", commandArgs(), value = TRUE)[1]))
if (length(HERE) == 0 || is.na(HERE) || HERE == "") HERE <- getwd()
SUBDIR   <- if (mode == "high_attn") "high_attn" else "all_cells"
OUT_DIR  <- file.path(HERE, "results_logfc", SUBDIR)
DE_DIR   <- file.path(HERE, "results", SUBDIR, "de_wilcox")
GMT_DIR  <- file.path(HERE, "..", "de_fgsea_results", "gmt")

GMT_PATHS <- list(
  c7  = file.path(GMT_DIR, "c7.all.v7.0.symbols.gmt"),
  h   = file.path(GMT_DIR, "h.all.v7.0.symbols.gmt"),
  c7h = file.path(GMT_DIR, "c7_h.all.v7.0.symbols.gmt")
)

cat(sprintf("R %s | fgsea %s\n", as.character(getRversion()),
            as.character(packageVersion("fgsea"))))
cat(sprintf("mode        : %s\n", mode))
cat(sprintf("collections : %s\n", paste(COLLECTIONS, collapse = ", ")))
cat(sprintf("DE dir      : %s\n", DE_DIR))
cat(sprintf("out dir     : %s\n", OUT_DIR))

# ── fgsea params ───────────────────────────────────────────────────────────
FGSEA_MIN <- 10        # C7 has many small sigs; 10 is the conventional floor
FGSEA_MAX <- 500
FGSEA_NP  <- 100000
SEED      <- 42
N_WORKERS <- as.integer(Sys.getenv("FGSEA_WORKERS", unset = "4"))
if (is.na(N_WORKERS) || N_WORKERS < 1) N_WORKERS <- 1L
set.seed(SEED)

# ── Worker ─────────────────────────────────────────────────────────────────
process_ct <- function(de_path, pathways, fgsea_dir) {
  ct  <- tools::file_path_sans_ext(basename(de_path))
  out_path <- file.path(fgsea_dir, paste0(ct, ".csv"))
  if (file.exists(out_path)) {
    cat(sprintf("[%s] skip (resume)\n", ct))
    return(tryCatch(fread(out_path), error = function(e) NULL))
  }

  de <- fread(de_path)
  rdf <- de[!is.na(logFC) & logFC != 0]
  if (nrow(rdf) < 20) {
    cat(sprintf("[%s] too few ranked genes; skip\n", ct))
    return(NULL)
  }
  # Rank by raw logFC (effect size). Positive = up in Inflamed.
  ranks  <- setNames(as.numeric(rdf$logFC), rdf$gene)
  ranks  <- ranks[!is.na(ranks)]
  ranks  <- ranks[order(-abs(ranks))]
  ranks  <- ranks[!duplicated(names(ranks))]
  ranks  <- sort(ranks, decreasing = TRUE)

  t0 <- Sys.time()
  fg <- tryCatch(
    fgseaSimple(pathways = pathways, stats = ranks,
                minSize = FGSEA_MIN, maxSize = FGSEA_MAX, nperm = FGSEA_NP,
                BPPARAM = SerialParam()),
    error = function(e) { cat(sprintf("[%s] ! fgsea failed: %s\n", ct, conditionMessage(e))); NULL }
  )
  if (is.null(fg) || nrow(fg) == 0) return(NULL)
  fg[, cell_type := ct]
  setorder(fg, padj, pval)
  fg_out <- fg[, .(pathway, pval, padj, ES, NES, size,
                   leadingEdge = sapply(leadingEdge, paste, collapse = ";"),
                   cell_type)]
  fwrite(fg_out, out_path)
  cat(sprintf("[%s] %d pathways tested, %d FDR<0.05  (%.1fs)\n",
              ct, nrow(fg_out), sum(fg_out$padj < 0.05, na.rm = TRUE),
              as.numeric(Sys.time() - t0, units = "secs")))
  fg_out
}

# ── Loop over collections ──────────────────────────────────────────────────
de_files <- list.files(DE_DIR, pattern = "\\.csv$", full.names = TRUE)
if (!length(de_files)) stop(sprintf("no DE CSVs in %s — run wilcox_de.py first", DE_DIR))

BPPARAM <- if (N_WORKERS > 1) MulticoreParam(workers = N_WORKERS, RNGseed = SEED, progressbar = FALSE) else SerialParam(RNGseed = SEED)

for (coll in COLLECTIONS) {
  gmt_path  <- GMT_PATHS[[coll]]
  fgsea_dir <- file.path(OUT_DIR, paste0("fgsea_", coll))
  dir.create(fgsea_dir, showWarnings = FALSE, recursive = TRUE)
  stopifnot(file.exists(gmt_path))
  pathways <- gmtPathways(gmt_path)
  cat(sprintf("\n================ collection: %s (%d sets) ================\n",
              coll, length(pathways)))
  cat(sprintf("GMT       : %s\n", gmt_path))
  cat(sprintf("fgsea dir : %s\n", fgsea_dir))
  cat(sprintf("processing %d cell types with %d worker(s) ...\n", length(de_files), N_WORKERS))

  results <- bplapply(de_files, process_ct,
                      pathways = pathways, fgsea_dir = fgsea_dir,
                      BPPARAM = BPPARAM)
  results <- results[!sapply(results, is.null)]

  # ── Combine ──────────────────────────────────────────────────────────────
  fg_files <- list.files(fgsea_dir, pattern = "\\.csv$", full.names = TRUE)
  if (length(fg_files)) {
    combo <- rbindlist(lapply(fg_files, fread), fill = TRUE)
    fwrite(combo, file.path(OUT_DIR, sprintf("fgsea_%s_combined.csv", coll)))

    combo[, signed_neglog10padj := sign(NES) * (-log10(pmax(padj, 1e-300)))]
    mat_pf <- dcast(combo, pathway ~ cell_type,
                    value.var = "signed_neglog10padj", fill = 0)
    fwrite(mat_pf, file.path(OUT_DIR, sprintf("fgsea_%s_signed_neglog10padj_matrix.csv", coll)))

    sig <- combo[, .(n_sig = sum(padj < 0.05, na.rm = TRUE),
                     n_up  = sum(padj < 0.05 & NES > 0, na.rm = TRUE),
                     n_dn  = sum(padj < 0.05 & NES < 0, na.rm = TRUE),
                     n_ct  = uniqueN(cell_type)),
                  by = pathway]
    sig[, frac_sig := n_sig / n_ct]
    setorder(sig, -frac_sig, -n_sig)
    fwrite(sig, file.path(OUT_DIR, sprintf("fgsea_%s_pathway_summary.csv", coll)))

    cat(sprintf("\n[%s] combined: %d (pathway,cell_type) rows from %d CTs\n",
                coll, nrow(combo), length(fg_files)))
    cat(sprintf("[%s] pathways significant (FDR<0.05) in >=10%% of cell types: %d\n",
                coll, sum(sig$frac_sig >= 0.10)))
  }
}

cat("\nDone.\n")
