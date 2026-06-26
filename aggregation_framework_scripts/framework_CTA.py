import torch
import torch.nn.functional as F
from torch import nn
from peft import LoraConfig, get_peft_model
from typing import Dict

from torch_geometric.utils import softmax as scatter_softmax
from torch_geometric.nn import global_add_pool, global_mean_pool

from scgpt.model import TransformerModel
from scgpt.utils import load_pretrained
from scgpt.tokenizer import GeneVocab


# ============================================================
# Cell-Type Attention (CTA) aggregator — Do et al. HA, attn1=False
# ============================================================
#
# Strict port of the model used in `cta_20seeds_3fold_cv.ipynb`, which is the
# same `HAModel` as the HA framework but configured with:
#     ATTN1 = False   # cell-level: MEAN pool cells within each cell-type group
#     ATTN2 = True    # cell-type-level: attention across cell types
#
# i.e. "Cell-Type Attention" only — there is no learned attention over
# individual cells; each cell-type group is mean-pooled, then a learned
# attention is applied across the per-cell-type representations within a
# patient.
#
# As in the HA framework, the notebook packs many patients into one flat batch
# using group indices `idx * n_ct + ct_dict[ct]`. The framework processes **one
# patient at a time**, so the per-cell group index is simply the cell-type index
# in `[0, n_ct)` and the pooling `size` is `n_ct`.


class HAModel(nn.Module):
    """
    Hierarchical Attention MIL head (single patient per forward).

    Identical architecture to the HA framework; CTA simply defaults
    ``attn1=False`` (mean-pool cells within each cell-type group) while keeping
    ``attn2=True`` (attention across cell types).

    Args:
        n_in         : int  — cell embedding dim (scGPT embsize / d_model)
        n_out        : int  — number of output classes
        n_ct         : int  — number of cell types (fixed for the dataset)
        attn1        : bool — enable cell-level attention (else mean pool)
        attn2        : bool — enable cell-type-level attention (else mean)
        dropout      : float
        n_layers_lin : int  — number of linear layers before attention
        n_hid        : int  — hidden dimension
    """

    def __init__(
        self,
        n_in: int,
        n_ct: int,
        n_out: int = 2,
        attn1: bool = False,
        attn2: bool = True,
        dropout: float = 0.0,
        n_layers_lin: int = 1,
        n_hid: int = 256,
    ):
        super().__init__()
        self.attn1 = attn1
        self.attn2 = attn2
        self.n_ct = n_ct

        self.lin = nn.Sequential(
            *self._build_layers(n_layers_lin, n_in, n_hid, n_hid, dropout)
        )
        curr_in = n_in if len(self.lin) == 0 else n_hid
        self.n_in1 = curr_in

        self.w_c = nn.Sequential(nn.Linear(curr_in, 1), nn.Dropout(dropout))
        self.w_ct = nn.Sequential(nn.Linear(curr_in, 1), nn.Dropout(dropout))
        self.lin_out = nn.Linear(curr_in, n_out)

    @staticmethod
    def _build_layers(n_layers, n_in, n_hid, n_out, dropout):
        layers = []
        for i in range(n_layers):
            c_in = n_in if i == 0 else n_hid
            c_out = n_out if i == n_layers - 1 else n_hid
            layers.extend([nn.Linear(c_in, c_out), nn.ReLU(), nn.Dropout(dropout)])
        return layers

    def forward(self, X: torch.Tensor, cell_types: torch.LongTensor):
        """
        Args:
            X          : (N_cells, n_in)  — cell embeddings for one patient
            cell_types : (N_cells,)       — per-cell group index in [0, n_ct)

        Returns:
            logits      : (1, n_out)   — raw class logits
            patient_emb : (1, n_in1)   — patient representation (pre-classifier)
            weights     : (1, N_cells) — level-1 per-cell attention (or None)
        """
        X = self.lin(X)

        if self.attn1:
            w_c = scatter_softmax(self.w_c(X).squeeze(-1), cell_types)  # (N_cells,)
            X = global_add_pool(X * w_c.unsqueeze(-1), cell_types, size=self.n_ct)
            weights = w_c.unsqueeze(0)
        else:
            X = global_mean_pool(X, cell_types, size=self.n_ct)
            weights = None

        X = X.reshape(-1, self.n_ct, self.n_in1)  # (1, n_ct, n_in1)

        if self.attn2:
            w_ct = torch.softmax(self.w_ct(X), dim=1)  # (1, n_ct, 1)
            X = torch.sum(X * w_ct, dim=1)             # (1, n_in1)
        else:
            X = torch.mean(X, dim=1)

        patient_emb = X
        logits = self.lin_out(X)
        return logits, patient_emb, weights


# ============================================================
# AggregatorPlusClassifier — the complete CTA head
# passed to SingleCellMetricModel as aggregator_plus_classifier
# ============================================================

class AggregatorPlusClassifier(nn.Module):
    """
    Cell-Type Attention aggregator + classifier.

    Wraps `HAModel` (with ``attn1=False`` by default) so it conforms to the
    framework aggregator contract: a single forward that maps a patient's cell
    embeddings to class logits.

    Args:
        d_model      : int   — cell embedding dimension (scGPT embsize)
        num_classes  : int   — number of disease classes
        n_cell_types : int   — number of cell types (fixed for the dataset)
        attn1        : bool  — cell-level attention (CTA default: False)
        attn2        : bool  — cell-type-level attention (CTA default: True)
        n_hid        : int   — hidden dimension
        n_layers_lin : int   — linear layers before attention
        dropout      : float
        normalize    : bool  — L2-normalise cell embeddings before pooling.
                               Defaults to False because the framework backbone
                               already L2-normalises cell embeddings, and the
                               notebook applied no internal normalisation.
    """

    def __init__(
        self,
        d_model: int,
        num_classes: int,
        n_cell_types: int,
        attn1: bool = False,
        attn2: bool = True,
        n_hid: int = 256,
        n_layers_lin: int = 1,
        dropout: float = 0.3,
        normalize: bool = False,
    ):
        super().__init__()
        self.normalize = normalize
        self.n_cell_types = n_cell_types
        self.ha = HAModel(
            n_in=d_model,
            n_ct=n_cell_types,
            n_out=num_classes,
            attn1=attn1,
            attn2=attn2,
            dropout=dropout,
            n_layers_lin=n_layers_lin,
            n_hid=n_hid,
        )

    def forward(self, bag_tensor: torch.Tensor, cell_types: torch.LongTensor):
        """
        Args:
            bag_tensor : (N_cells, d_model)
            cell_types : (N_cells,) — per-cell cell-type index in [0, n_cell_types)

        Returns:
            preds       : (1, num_classes) — raw logits
            patient_emb : (1, n_in1)       — patient representation (pre-classifier)
            weights     : (1, N_cells)     — level-1 per-cell attention (None for CTA)
        """
        if self.normalize:
            bag_tensor = F.normalize(bag_tensor, dim=-1)
        return self.ha(bag_tensor, cell_types)


# ============================================================
# Factory function — builds aggregator from model_config
# ============================================================

def build_aggregator(
    emb_size: int,
    num_classes: int,
    n_cell_types: int,
    attn1: bool = False,
    attn2: bool = True,
    n_hid: int = 256,
    n_layers_lin: int = 1,
    dropout: float = 0.3,
    normalize: bool = False,
) -> AggregatorPlusClassifier:
    """
    Build the Cell-Type Attention aggregator + classifier head.

    Args:
        emb_size     : int  — cell embedding dim (scGPT embsize / d_model)
        num_classes  : int  — number of output classes
        n_cell_types : int  — number of cell types (fixed for the dataset)
        attn1        : bool — cell-level attention (CTA default: False)
        attn2        : bool — cell-type-level attention (CTA default: True)
        n_hid        : int  — hidden dimension
        n_layers_lin : int  — linear layers before attention
        dropout      : float
        normalize    : bool

    Returns:
        AggregatorPlusClassifier instance
    """
    return AggregatorPlusClassifier(
        d_model=emb_size,
        num_classes=num_classes,
        n_cell_types=n_cell_types,
        attn1=attn1,
        attn2=attn2,
        n_hid=n_hid,
        n_layers_lin=n_layers_lin,
        dropout=dropout,
        normalize=normalize,
    )


# ============================================================
# SingleCellMetricModel — scGPT backbone + LoRA + CTA aggregator
# ============================================================

class SingleCellMetricModel(nn.Module):
    def __init__(
        self,
        model_config: Dict,
        checkpoint_path: str,
        vocab: GeneVocab,
        aggregator_plus_classifier: nn.Module,
        lora_r: int = 8,
        max_seq_length: int = 1200,
        device: str = "cpu"
    ):
        super().__init__()

        self.device_ = torch.device(device)

        # Store special token IDs and pad_value for use in forward
        self.cls_token_id = vocab["<cls>"]
        self.pad_token_id = vocab[model_config["pad_token"]]
        self.pad_value = model_config["pad_value"]
        self.max_seq_length = max_seq_length

        self.backbone = TransformerModel(
            ntoken=len(vocab),
            d_model=model_config["embsize"],
            nhead=model_config["nheads"],
            d_hid=model_config["d_hid"],
            nlayers=model_config["nlayers"],
            nlayers_cls=model_config["n_layers_cls"],
            vocab=vocab,
            dropout=model_config["dropout"],
            pad_token=model_config["pad_token"],
            pad_value=model_config["pad_value"],
            do_mvc=True,
            do_dab=False,
            use_batch_labels=False,
            domain_spec_batchnorm=False,
            explicit_zero_prob=False,
            use_fast_transformer=False,
            pre_norm=False,
        )

        # Load Pretrained Weights
        print(f"Loading weights from {checkpoint_path}...")
        load_pretrained(self.backbone, torch.load(checkpoint_path, map_location='cpu'), verbose=False)

        # Inject LoRA Adapters
        lora_config = LoraConfig(
            r=lora_r,
            lora_alpha=lora_r * 2,
            target_modules=["out_proj"],
            # target_modules=["in_proj", "out_proj", "linear1", "linear2"],
            lora_dropout=0.05,
            bias="none",
            task_type="SEQ_CLS"
        )
        self.backbone = get_peft_model(self.backbone, lora_config)

        self.aggregator_plus_classifier = aggregator_plus_classifier

        # Move entire model to the target device
        self.to(self.device_)

    def _sparsify_and_collate(
        self,
        tokens: torch.LongTensor,
        expression_chunk: torch.Tensor,
        max_length: int,
    ):
        """
        Per-cell sparse tokenization matching scGPT's pretraining format.

        For each cell:
          1. Extract only non-zero expression positions
          2. Gather corresponding gene tokens and expression values
          3. Prepend <cls> token with pad_value expression
          4. If length > max_length: randomly sample (keeping CLS at position 0)
          5. Pad shorter sequences to batch max length

        Args:
            tokens: (Genes,) shared gene token IDs for all cells
            expression_chunk: (chunk_cells, Genes) binned expression values
            max_length: maximum sequence length (1200 for scGPT)

        Returns:
            gene_batch: (chunk_cells, padded_len)
            expr_batch: (chunk_cells, padded_len)
            padding_mask: (chunk_cells, padded_len) True at padding positions
        """
        device = tokens.device
        n_cells = expression_chunk.size(0)

        cell_genes_list = []
        cell_exprs_list = []

        for i in range(n_cells):
            row = expression_chunk[i]  # (Genes,)

            # Extract non-zero positions (sparse representation)
            nonzero_idx = row.nonzero(as_tuple=True)[0]

            if len(nonzero_idx) == 0:
                # Edge case: cell with no expressed genes — just CLS
                cell_genes = torch.tensor([self.cls_token_id], dtype=tokens.dtype, device=device)
                cell_exprs = torch.tensor([self.pad_value], dtype=torch.float32, device=device)
            else:
                cell_genes = tokens[nonzero_idx]
                cell_exprs = row[nonzero_idx].float()

                # Prepend <cls> token at position 0 with pad_value as expression
                cell_genes = torch.cat([
                    torch.tensor([self.cls_token_id], dtype=tokens.dtype, device=device),
                    cell_genes
                ])
                cell_exprs = torch.cat([
                    torch.tensor([self.pad_value], dtype=torch.float32, device=device),
                    cell_exprs
                ])

                # Random sample if too long (keep CLS at position 0)
                if len(cell_genes) > max_length:
                    perm = torch.randperm(len(cell_genes) - 1, device=device)[:max_length - 1]
                    indices = torch.cat([
                        torch.zeros(1, dtype=torch.long, device=device),
                        perm + 1
                    ])
                    cell_genes = cell_genes[indices]
                    cell_exprs = cell_exprs[indices]

            cell_genes_list.append(cell_genes)
            cell_exprs_list.append(cell_exprs)

        # Determine padded length (min of batch max and configured max_length)
        max_len_in_chunk = max(len(g) for g in cell_genes_list)
        padded_len = min(max_len_in_chunk, max_length)

        # Pad and stack into batch tensors
        gene_batch = torch.full(
            (n_cells, padded_len), self.pad_token_id, dtype=tokens.dtype, device=device
        )
        expr_batch = torch.full(
            (n_cells, padded_len), self.pad_value, dtype=torch.float32, device=device
        )

        for i in range(n_cells):
            seq_len = len(cell_genes_list[i])
            gene_batch[i, :seq_len] = cell_genes_list[i]
            expr_batch[i, :seq_len] = cell_exprs_list[i]

        # Padding mask: True where gene_id == pad_token_id (scGPT convention)
        padding_mask = gene_batch.eq(self.pad_token_id)

        return gene_batch, expr_batch, padding_mask

    def forward(self, batch: Dict[str, torch.Tensor], chunk_size: int = 512):
        """
        Memory-efficient forward pass for large cell counts.
        Accepts a batch dict from CustomDataset / DataLoader (batch_size=1).

        Matches scGPT embedding convention:
        - Only non-zero expressed genes are tokenized per cell (sparse)
        - Sequences capped at max_seq_length (default 1200)
        - CLS token at position 0 used as cell embedding
        - L2 normalization applied to cell embeddings

        Required batch keys:
        - "tokens"     : (Genes,)        shared gene token IDs
        - "expression" : (Cells, Genes)  binned expression values
        - "cell_types" : (Cells,)        per-cell cell-type index in [0, n_cell_types)
        """
        tokens = batch["tokens"].squeeze(0).to(self.device_)          # (Genes,)
        expression = batch["expression"].squeeze(0).to(self.device_)  # (Cells, Genes)
        cell_types = batch["cell_types"].squeeze(0).to(self.device_).long()  # (Cells,)

        n_cells = expression.size(0)
        cell_embs = []

        # Process cells in chunks to avoid GPU OOM while keeping graph active
        for i in range(0, n_cells, chunk_size):
            chunk_expr = expression[i : i + chunk_size]

            # Per-cell sparse extraction + collation (matches scGPT Dataset + DataCollator)
            gene_batch, expr_batch, padding_mask = self._sparsify_and_collate(
                tokens, chunk_expr, self.max_seq_length
            )

            # Encode through transformer backbone
            output = self.backbone._encode(
                gene_batch,
                expr_batch,
                src_key_padding_mask=padding_mask,
            )

            # Extract CLS token (position 0) as cell embedding
            cell_embs.append(output[:, 0, :])

        # Concatenate cell embeddings to build the Patient-Bag representation
        bag_tensor = torch.cat(cell_embs, dim=0)  # (N_cells, d_model)

        # L2 normalize embeddings (matches reference scGPT cell_emb.py)
        bag_tensor = F.normalize(bag_tensor, p=2, dim=-1)

        # Aggregate + classify via the externally-supplied CTA head
        return self.aggregator_plus_classifier(bag_tensor, cell_types)
