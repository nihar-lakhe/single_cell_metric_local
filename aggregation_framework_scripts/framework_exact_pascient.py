import torch
import torch.nn.functional as F
from torch import nn
from peft import LoraConfig, get_peft_model
from typing import Dict

from scgpt.model import TransformerModel
from scgpt.utils import load_pretrained
from scgpt.tokenizer import GeneVocab


# ============================================================
# Attention Pooling — PaSCient `LinearAttnAggregator` style
# ============================================================
#
# Strict port of `pascient.components.aggregators.LinearAttnAggregator`:
#     sm = softmax(Z, dim=cells)        # softmax of the embedding values
#     patient_emb = (sm * Z).sum(cells)  # per-feature attention
#
# No scorer MLP, no Tanh, no learnable parameters in the aggregator.

class AttentionPooling(nn.Module):
    """
    PaSCient `LinearAttnAggregator` style pooling — parameter-free.

    Softmax is applied **per-feature** across the cells dimension (each of the
    d_model channels gets its own softmax distribution over cells), then the
    cells are summed. This matches the binary-experiment aggregator exactly.

    Input:
        Z : (N_cells, d_model) — one patient at a time (no batch dim)

    Output:
        patient_emb : (1, d_model)
        weights     : (1, N_cells) — per-cell attention weight, summarised by
                                     averaging the per-feature softmax over the
                                     d_model channels (provided for downstream
                                     interpretability — not used in the forward
                                     computation, which uses the full per-feature
                                     softmax).
    """

    def __init__(self):
        super().__init__()

    def forward(self, Z: torch.Tensor):
        sm = F.softmax(Z, dim=0)                         # (N_cells, d_model)
        patient_emb = (sm * Z).sum(dim=0, keepdim=True)  # (1, d_model)
        weights = sm.mean(dim=-1).unsqueeze(0)           # (1, N_cells) — summary
        return patient_emb, weights


# ============================================================
# Patient encoder — PaSCient `patient_encoder` style
# ============================================================
#
# Strict port of patient_encoder used by the binary experiment:
#   BasicMLP(input_dim=d_model, hidden_dim=d_model//2, output_dim=d_model//2,
#            n_hidden_layers=0, activation_cls=PReLU, activation_out_cls=PReLU)
# Layout: Linear → PReLU → Linear → PReLU   (no dropout)

class PatientEncoder(nn.Module):
    """
    Post-pool patient encoder MLP with PReLU activations.

    Args:
        d_model : int — input dim (patient embedding dim out of pooling)
        enc_hid : int — hidden / output dim (PaSCient default: d_model // 2)
    """

    def __init__(self, d_model: int, enc_hid: int):
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Linear(d_model, enc_hid),
            nn.PReLU(),
            nn.Linear(enc_hid, enc_hid),
            nn.PReLU(),
        )

    def forward(self, patient_emb: torch.Tensor):
        return self.encoder(patient_emb)


# ============================================================
# Patient classifier — PaSCient `patient_predictor` style
# ============================================================
#
# Mirrors `patient_predictor = BasicMLP(enc_hid, num_classes, num_classes,
#                                       n_hidden_layers=-1)`
# which collapses to a single Linear(enc_hid → num_classes).

class PatientClassifier(nn.Module):
    """
    Single-Linear patient classifier (PaSCient `patient_predictor` style).

    Args:
        enc_hid     : int — input dim (output of PatientEncoder)
        num_classes : int — number of output classes
    """

    def __init__(self, enc_hid: int, num_classes: int):
        super().__init__()
        self.classifier = nn.Linear(enc_hid, num_classes)

    def forward(self, patient_feat: torch.Tensor):
        return self.classifier(patient_feat)


# ============================================================
# AggregatorPlusClassifier — the complete module
# passed to SingleCellMetricModel as aggregator_plus_classifier
# ============================================================

class AggregatorPlusClassifier(nn.Module):
    """
    Strict PaSCient-binary head:
       LinearAttnAggregator → PReLU patient_encoder (d → d/2 → d/2) → Linear classifier.

    Accepts bag_tensor of shape (N_cells, d_model) and returns:
        preds       : (1, num_classes) — logits for loss computation
        patient_emb : (1, d_model)     — pooled patient representation (pre-encoder)
        weights     : (1, N_cells)     — per-cell attention summary (mean over features)

    Args:
        d_model     : int   — cell embedding dimension (must match scGPT embsize)
        num_classes : int   — number of disease classes
        enc_hid     : int   — patient_encoder hidden / output dim
                              (PaSCient default: d_model // 2)
        normalize   : bool  — L2-normalise cell embeddings before pooling
    """

    def __init__(
        self,
        d_model: int,
        num_classes: int,
        enc_hid: int = None,
        normalize: bool = True,
    ):
        super().__init__()

        if enc_hid is None:
            enc_hid = d_model // 2

        self.normalize = normalize
        self.attention_pooling = AttentionPooling()
        self.patient_encoder = PatientEncoder(d_model=d_model, enc_hid=enc_hid)
        self.classifier = PatientClassifier(enc_hid=enc_hid, num_classes=num_classes)

    def forward(self, bag_tensor: torch.Tensor):
        """
        Args:
            bag_tensor : (N_cells, d_model)

        Returns:
            preds       : (1, num_classes) — raw logits
            patient_emb : (1, d_model)     — pooled (pre-encoder) patient vector
            weights     : (1, N_cells)     — per-cell attention summary
        """
        if self.normalize:
            bag_tensor = F.normalize(bag_tensor, dim=-1)

        patient_emb, weights = self.attention_pooling(bag_tensor)
        patient_feat = self.patient_encoder(patient_emb)
        preds = self.classifier(patient_feat)

        return preds, patient_emb, weights


# ============================================================
# Factory function — builds aggregator from model_config
# ============================================================

def build_aggregator(
    emb_size: int,
    num_classes: int,
    enc_hid: int = None,
    normalize: bool = True,
) -> AggregatorPlusClassifier:
    """
    Build the strict PaSCient-binary aggregator + classifier head.

    Args:
        emb_size    : int  — cell embedding dim (scGPT embsize / d_model)
        num_classes : int  — number of output classes
        enc_hid     : int  — patient_encoder hidden / output dim
                             (default: emb_size // 2, matching PaSCient)
        normalize   : bool

    Returns:
        AggregatorPlusClassifier instance
    """
    return AggregatorPlusClassifier(
        d_model=emb_size,
        num_classes=num_classes,
        enc_hid=enc_hid,
        normalize=normalize,
    )


# ============================================================
# SingleCellMetricModel — scGPT backbone + LoRA + aggregator
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
        """
        tokens = batch["tokens"].squeeze(0).to(self.device_)         # (Genes,)
        expression = batch["expression"].squeeze(0).to(self.device_)  # (Cells, Genes)

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

        # Aggregate + classify via the externally-supplied head
        return self.aggregator_plus_classifier(bag_tensor)