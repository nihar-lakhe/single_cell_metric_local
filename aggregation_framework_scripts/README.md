# Framework Scripts

Model-definition modules that pair the **scGPT** backbone (with LoRA adapters)
to a patient-level **aggregator + classifier** head. Each script exposes the
same contract so they are drop-in interchangeable in a training driver:

```python
from framework_<name> import SingleCellMetricModel, build_aggregator
```

| Script | Aggregator head | Reference |
| --- | --- | --- |
| `framework_exact_pascient.py` | PaSCient `LinearAttnAggregator` (parameter-free per-feature softmax pooling) + PReLU encoder + Linear classifier | binary PaSCient experiment |
| `framework_HA.py` | Hierarchical Attention (HA) — cell-level **and** cell-type-level attention (`attn1=True, attn2=True`) | `ha_20seeds_3fold_cv.ipynb` |
| `framework_CTA.py` | Cell-Type Attention (CTA) — mean-pool cells, cell-type-level attention only (`attn1=False, attn2=True`) | `cta_20seeds_3fold_cv.ipynb` |
| `framework_pascient_ha_like.py` | Softmax Attention Pooling (SAP) — HA-mirrored single-level cell→patient softmax attention (no cell-type hierarchy) + single Linear classifier | `sap_ha_mirrored_20seeds_3fold_cv.ipynb` |

---

## Shared contract

Both scripts provide:

- **`build_aggregator(...)`** — factory that returns the aggregator + classifier
  head (`AggregatorPlusClassifier`).
- **`SingleCellMetricModel(model_config, checkpoint_path, vocab, aggregator_plus_classifier, lora_r, max_seq_length, device)`**
  — scGPT `TransformerModel` backbone, pretrained weights loaded via
  `load_pretrained`, LoRA injected on `out_proj`, plus the supplied head.

`SingleCellMetricModel.forward(batch, chunk_size)` returns a 3-tuple:

```python
preds, patient_emb, weights = model(batch, chunk_size=512)
# preds : (1, num_classes) raw logits  -> feed to CrossEntropyLoss
```

The backbone processes **one patient (bag of cells) per forward**, in chunks to
bound GPU memory:

1. Per-cell sparse tokenization matching scGPT pretraining (non-zero genes only,
   `<cls>` prepended, capped at `max_seq_length`, default 1200).
2. CLS token (position 0) taken as each cell embedding.
3. Cell embeddings L2-normalized, concatenated into `bag_tensor (N_cells, d_model)`.
4. `bag_tensor` handed to the aggregator head.

---

## `framework_HA.py` — Hierarchical Attention

Strict port of the `HAModel` from `ha_20seeds_3fold_cv.ipynb`, adapted to the
framework's **one-patient-per-forward** flow.

### How it works

- **Level 1 (`attn1`)** — attention over cells *within* each cell-type group
  (`torch_geometric.softmax` + `global_add_pool`), producing one vector per cell
  type.
- **Level 2 (`attn2`)** — attention over the per-cell-type vectors *within* the
  patient (plain softmax + weighted sum), producing the patient representation.
- A `Linear` classifier maps the patient representation to class logits.

In the notebook, many patients are packed into one flat batch via the group
index `idx * n_ct + ct_dict[ct]` with pool `size = n_ct * n_patients`. Because
the framework runs a single patient at a time, that index collapses to just the
**cell-type index** in `[0, n_ct)` and the pool `size` is `n_ct`. Cell types
absent for a patient yield an all-zero row from `global_add_pool` (size=`n_ct`);
those zero rows still participate in the level-2 softmax exactly as in the
notebook. This single-patient head was verified to produce **bit-identical**
output to the notebook `HAModel` (max abs diff = 0.0).

### Building the head

```python
aggregator = build_aggregator(
    emb_size=model_config["embsize"],  # scGPT d_model (e.g. 512)
    num_classes=2,
    n_cell_types=N_CT,                 # fixed number of cell types in the dataset
    attn1=True,                        # cell-level attention
    attn2=True,                        # cell-type-level attention
    n_hid=256,                         # notebook N_HID
    n_layers_lin=1,                    # notebook N_LAYERS_LIN
    dropout=0.3,                       # notebook DROPOUT
    normalize=False,                   # see note below
)

model = SingleCellMetricModel(
    model_config=model_config,
    checkpoint_path=checkpoint_path,
    vocab=vocab,
    aggregator_plus_classifier=aggregator,
    lora_r=8,
    max_seq_length=1200,
    device="cuda",
)
```

Defaults match the notebook hyperparameters
(`N_HID=256`, `N_LAYERS_LIN=1`, `DROPOUT=0.3`, `ATTN1=ATTN2=True`).

### Required batch keys

`SingleCellMetricModel.forward` (HA variant) expects the batch dict to contain:

| Key | Shape | Description |
| --- | --- | --- |
| `tokens` | `(Genes,)` | shared gene token IDs |
| `expression` | `(Cells, Genes)` | binned expression values |
| `cell_types` | `(Cells,)` | **per-cell cell-type index in `[0, n_cell_types)`** |
| `label` | scalar | patient label (used by the training driver) |

> **`cell_types` must be in the same row order as `expression`.** The current
> `CustomDataset` does **not** yet emit `cell_types` — the dataset must be
> updated to provide it (map the `Celltype` annotation through a fixed
> `{cell_type: index}` dictionary), and `n_cell_types` passed to
> `build_aggregator` must equal that dictionary's size.

### Returns

```python
preds, patient_emb, weights = model(batch)
# preds       : (1, num_classes)  raw logits
# patient_emb : (1, n_in1)        patient representation (pre-classifier)
# weights     : (1, N_cells)      level-1 per-cell attention (None if attn1=False)
```

### Notes / differences vs. PaSCient framework

- **Cell-type dependency** — unlike the PaSCient head (which only needs
  `bag_tensor`), the HA head additionally needs per-cell cell-type indices,
  hence the extra `cell_types` batch key and the `n_cell_types` build argument.
- **Single normalization** — the backbone L2-normalizes cell embeddings once;
  the HA head's `normalize` defaults to `False` (the notebook applied none),
  avoiding the redundant double-normalize present in the PaSCient head.
- **Changed `build_aggregator` signature** — it now requires `n_cell_types`
  (and accepts `attn1/attn2/n_hid/n_layers_lin/dropout`), so a driver written
  for `framework_exact_pascient.py` must update this call when switching to HA.

### Dependencies

In addition to the scGPT / PEFT stack used by the other framework scripts, the
HA head requires **`torch_geometric`** (`softmax`, `global_add_pool`,
`global_mean_pool`).

---

## `framework_CTA.py` — Cell-Type Attention

Strict port of the model in `cta_20seeds_3fold_cv.ipynb`. This is the **same
`HAModel`** as `framework_HA.py`, configured with `attn1=False, attn2=True`:

- **Level 1 (`attn1=False`)** — cells within each cell-type group are
  **mean-pooled** (no learned attention over individual cells).
- **Level 2 (`attn2=True`)** — learned attention across the per-cell-type
  representations within the patient.
- A `Linear` classifier maps the patient representation to class logits.

In other words: "Cell-Type Attention" only. Verified to produce
**bit-identical** output to the notebook for a single patient (max abs diff = 0.0).

### Building the head

```python
from framework_CTA import SingleCellMetricModel, build_aggregator

aggregator = build_aggregator(
    emb_size=model_config["embsize"],
    num_classes=2,
    n_cell_types=N_CT,
    attn1=False,   # CTA default — mean-pool cells
    attn2=True,    # CTA default — attention across cell types
    n_hid=256,
    n_layers_lin=1,
    dropout=0.3,
    normalize=False,
)
```

### Interface

Identical to `framework_HA.py` in every other respect — same required batch keys
(`tokens`, `expression`, **`cell_types`**, `label`), same
`(preds, patient_emb, weights)` return tuple, same `torch_geometric` dependency,
and the same dataset caveat (the dataset must emit per-cell `cell_types`). The
only difference is the `attn1=False` default. Because `attn1=False`, the returned
`weights` is `None` (there is no cell-level attention to report).

---

## `framework_pascient_ha_like.py` — HA-mirrored Softmax Attention Pooling (SAP)

Strict port of the `SAPModel` from `sap_ha_mirrored_20seeds_3fold_cv.ipynb`.
This is **HA-style architecture with the cell-type hierarchy removed**: a
single-level learned softmax attention over a patient's cells, followed by a
single `Linear` classifier (which preserves exact per-cell logit decomposition).

### How it works

```
cells (d_model) ─▶ Linear(d_model, n_hid) ─▶ ReLU ─▶ Dropout
                ─▶ [softmax attention pooling over a patient's cells]
                ─▶ patient_emb (n_hid)
                ─▶ Linear(n_hid, num_classes) ─▶ logits
```

- **Attention (`attn=True`)** — per-cell scores via `Linear(n_hid, 1)`, softmax
  normalised across the patient's cells, then weighted sum into `patient_emb`.
- **Mean fallback (`attn=False`)** — cells are simple mean-pooled.

In the notebook, many patients are packed into one flat batch and a segmented
softmax (`torch_geometric.softmax(..., batch)`) is used to normalise per
patient. Because the framework runs **one patient at a time**, the `batch`
index is all zeros and the segmented softmax collapses to plain
`torch.softmax(scores, dim=0)`; `global_add_pool` collapses to `sum`.
The single-patient head is behaviourally identical to the notebook on a
single patient.

### Building the head

```python
from framework_pascient_ha_like import SingleCellMetricModel, build_aggregator

aggregator = build_aggregator(
    emb_size=model_config["embsize"],
    num_classes=2,
    attn=True,                 # cell-level attention (else mean pool)
    n_hid=256,                 # notebook N_HID
    n_layers_lin=1,            # notebook N_LAYERS_LIN
    dropout=0.3,               # notebook DROPOUT
    normalize=False,           # backbone already L2-normalises
)

model = SingleCellMetricModel(
    model_config=model_config,
    checkpoint_path=checkpoint_path,
    vocab=vocab,
    aggregator_plus_classifier=aggregator,
    lora_r=8,
    max_seq_length=1200,
    device="cuda",
)
```

Defaults match the notebook hyperparameters
(`N_HID=256`, `N_LAYERS_LIN=1`, `DROPOUT=0.3`, `ATTN=True`).

### Required batch keys

Same minimal interface as `framework_exact_pascient.py` — **no `cell_types`
needed**:

| Key | Shape | Description |
| --- | --- | --- |
| `tokens` | `(Genes,)` | shared gene token IDs |
| `expression` | `(Cells, Genes)` | binned expression values |
| `label` | scalar | patient label (used by the training driver) |

### Returns

```python
preds, patient_emb, weights = model(batch)
# preds       : (1, num_classes)  raw logits
# patient_emb : (1, n_hid)        patient representation (pre-classifier)
# weights     : (1, N_cells)      per-cell attention (None if attn=False)
```

### Notes / differences vs. other heads

- **vs `framework_HA.py` / `framework_CTA.py`** — no cell-type hierarchy:
  drops Level-2 attention entirely. Consequently, no `cell_types` batch key
  and **no `torch_geometric` dependency**.
- **vs `framework_exact_pascient.py`** — same minimal batch interface, but
  the head is a *learned* attention scorer (`Linear(n_hid, 1)` + softmax over
  cells) preceded by an MLP projection, instead of PaSCient's parameter-free
  per-feature softmax pooling + PReLU encoder.
- **Single Linear classifier head** — preserves exact per-cell logit
  decomposition (the property highlighted in the SAP notebook), unlike a
  multi-layer MLP head.
- **Single normalization** — the backbone L2-normalises cell embeddings once;
  `normalize` defaults to `False` (the notebook applied none), avoiding a
  redundant double-normalize.

### Dependencies

Same as `framework_exact_pascient.py` — only the scGPT / PEFT stack. **No
`torch_geometric` required.**
