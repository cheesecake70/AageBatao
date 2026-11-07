"""M1 word LSTM / M2 char LSTM: the same class, config-driven (PRD section 7).

    nn.Embedding(V, E) -> nn.LSTM(E, H, L, dropout=p) -> nn.Linear(H, V), tied when E == H.

After lstm_numpy.py this is a restatement; the trainer (train/trainer.py) owns the loop.

TODO 28 -> 30 live here. Tests: tests/test_lstm.py
"""

import numpy as np
import torch
from torch import nn


class WordLSTM(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        emb: int,
        hidden: int,
        layers: int,
        dropout: float,
        tie_weights: bool = True,
    ):
        super().__init__()
        if tie_weights and emb != hidden:
            raise ValueError(f"tie_weights needs emb == hidden; got emb={emb}, hidden={hidden}")
        self.embed = nn.Embedding(vocab_size, emb)  # word id -> E-vector (the NumPy `emb`)
        self.drop = nn.Dropout(dropout)  # applied to embeddings and to LSTM outputs
        self.lstm = nn.LSTM(
            emb,
            hidden,
            num_layers=layers,
            batch_first=True,  # tensors are [B, T, *], not torch's default [T, B, *]
            dropout=dropout if layers > 1 else 0.0,  # torch warns if dropout>0 with 1 layer
        )
        self.out = nn.Linear(hidden, vocab_size)  # h_t -> one logit per vocab word
        self.tied = tie_weights
        if tie_weights:
            # input and output embeddings share ONE matrix: fewer params, usually lower ppl
            self.out.weight = self.embed.weight

    def forward(
        self, x: torch.Tensor, state: tuple[torch.Tensor, torch.Tensor] | None = None
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        # TODO 28b: x is [B, T] long. Return (logits [B, T, V], (h, c)).
        emb = self.drop(self.embed(x))  # string to embedding
        output, (h, c) = self.lstm(emb, state)  # hidden state, cell state
        output = self.drop(output)
        logits = self.out(output)
        return logits, (h, c)

    @torch.no_grad()
    def log_probs(self, ids: list[int]) -> np.ndarray:
        # TODO 29: eval interface (see eval/evaluate.py). self.eval(); run ids[:-1] as a [1, T]
        #   batch through forward, log_softmax over the last dim, return [T, V] float64 numpy.
        #   (Run on the model's device: next(self.parameters()).device.)
        self.eval()
        device = next(self.parameters()).device

        x = torch.tensor(ids[:-1], dtype=torch.long, device=device).unsqueeze(0)

        logits, _ = self.forward(x)

        log_probs = torch.log_softmax(logits, dim=-1)

        return log_probs.squeeze(0).cpu().numpy().astype(np.float64)

    def count_params(self) -> tuple[int, int]:
        """Return (total, non_embedding).

        `non_embedding` is the recurrent core: everything except the two V-sized lookup
        matrices (input embedding and output projection). It is the x-axis of the PRD's scaling
        plot, because the 30k-way matrices otherwise swamp the hidden-size effect, and it must
        come out the same whether or not the two matrices are tied.
        """
        # parameters() yields each distinct tensor once, so a tied weight is counted once here
        total = sum(p.numel() for p in self.parameters())
        non_embedding = total - self.embed.weight.numel()  # drop the input embedding
        if not self.tied:
            non_embedding -= self.out.weight.numel()  # untied: the output matrix is separate too
        # (tied: the shared matrix was already removed by the line above; the output bias stays)
        return total, non_embedding


def from_config(cfg: dict, vocab_size: int) -> WordLSTM:
    """Build a WordLSTM from a configs/model/lstm_<size>.yaml dict.

    The yaml also carries training keys (batch_size, lr, ...), so pick the constructor's five
    explicitly instead of `**cfg`; a missing key raises KeyError on purpose.
    """
    return WordLSTM(
        vocab_size=vocab_size,
        emb=cfg["emb"],
        hidden=cfg["hidden"],
        layers=cfg["layers"],
        dropout=cfg["dropout"],
        tie_weights=cfg["tie_weights"],
    )
