"""Training loop for the torch models (PRD section 7): cross-entropy, teacher forcing,
truncated BPTT over windows, grad clipping 1.0, AdamW, warmup + cosine LR, early stopping on
val perplexity with patience 3. MLflow logging arrives in P2; P1 logs a csv per run.

TODO 31 -> 35 live here. Tests: tests/test_trainer.py
"""

import csv
import math
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader

ARTIFACTS_DIR = Path("artifacts")
DEFAULTS = {
    "lr": 1e-3,
    "weight_decay": 0.01,
    "warmup_steps": 1000,
    "clip": 1.0,
    "max_epochs": 30,
    "patience": 3,
}


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def make_scheduler(
    optimizer: torch.optim.Optimizer, warmup_steps: int, total_steps: int
) -> LambdaLR:
    def multiplier(step: int) -> float:
        if warmup_steps > 0 and step < warmup_steps:
            return step / warmup_steps  # linear warmup from 0
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        progress = min(1.0, max(0.0, progress))  # never below 0 or past the end
        return 0.5 * (1.0 + math.cos(math.pi * progress))  # cosine decay to 0

    return LambdaLR(optimizer, multiplier)


class Trainer:
    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        run_name: str,
        cfg: dict | None = None,
        artifacts_dir: Path = ARTIFACTS_DIR,
        device: str | None = None,
    ):
        self.cfg = {**DEFAULTS, **(cfg or {})}
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model.to(self.device)
        self.train_loader, self.val_loader = train_loader, val_loader
        self.run_name, self.artifacts_dir = run_name, Path(artifacts_dir)
        self.optimizer = torch.optim.AdamW(
            model.parameters(), lr=self.cfg["lr"], weight_decay=self.cfg["weight_decay"]
        )
        total = self.cfg["max_epochs"] * len(train_loader)
        # PRD says 1k warmup steps; cap at one epoch so tiny (smoke) runs are not all warmup
        warmup = min(self.cfg["warmup_steps"], len(train_loader))
        self.scheduler = make_scheduler(self.optimizer, warmup, total)
        self.loss_fn = nn.CrossEntropyLoss()
        self.history: list[dict] = []
        self.step = 0

    @staticmethod
    def _detach(state):
        # Truncated BPTT: keep the hidden state's VALUES across windows, cut the gradient graph.
        return None if state is None else tuple(s.detach() for s in state)

    def train_epoch(self) -> float:
        self.model.train()
        total, n_batches = 0.0, 0
        state = None  # DECISION: carry hidden state across (shuffled) windows, detached each step.
        for x, y in self.train_loader:
            x, y = x.to(self.device), y.to(self.device)
            logits, state = self.model(
                x, state
            )  # teacher forcing: inputs are the TRUE previous tokens
            state = self._detach(state)  # truncated BPTT: values flow on, gradients stop here
            loss = self.loss_fn(logits.reshape(-1, logits.shape[-1]), y.reshape(-1))
            self.optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg["clip"])  # explode guard
            self.optimizer.step()
            self.scheduler.step()
            self.step += 1
            total += loss.item()
            n_batches += 1
        return total / max(1, n_batches)

    @torch.no_grad()
    def evaluate_val(self) -> float:
        self.model.eval()
        total, n_tokens = 0.0, 0
        for x, y in self.val_loader:
            x, y = x.to(self.device), y.to(self.device)
            logits, _ = self.model(x, None)  # fresh state per batch: windows are independent here
            # sum (not mean) so the final perplexity is token-weighted across batches
            total += nn.functional.cross_entropy(
                logits.reshape(-1, logits.shape[-1]), y.reshape(-1), reduction="sum"
            ).item()
            n_tokens += y.numel()
        return math.exp(total / max(1, n_tokens))

    def fit(self) -> list[dict]:
        best, bad = math.inf, 0
        for epoch in range(1, self.cfg["max_epochs"] + 1):
            t0 = time.time()
            train_loss = self.train_epoch()
            val_ppl = self.evaluate_val()
            row = {
                "epoch": epoch,
                "step": self.step,
                "train_loss": round(train_loss, 4),
                "train_ppl": round(math.exp(train_loss), 2),
                "val_ppl": round(val_ppl, 2),
                "lr": self.scheduler.get_last_lr()[0],
                "seconds": round(time.time() - t0, 1),
            }
            self.history.append(row)
            print(
                f"[{self.run_name}] epoch {epoch:3d}  train_loss {train_loss:.4f}  "
                f"val_ppl {val_ppl:9.2f}  lr {row['lr']:.2e}  {row['seconds']:.0f}s",
                flush=True,
            )
            if val_ppl < best:
                best, bad = val_ppl, 0
                self.save_checkpoint()
            else:
                bad += 1
                if bad >= self.cfg["patience"]:
                    print(f"early stop: no val improvement for {bad} epochs", flush=True)
                    break
        self._write_log()
        return self.history

    def _write_log(self) -> Path:
        self.artifacts_dir.mkdir(parents=True, exist_ok=True)
        path = self.artifacts_dir / f"{self.run_name}_log.csv"
        if self.history:
            with path.open("w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=list(self.history[0]))
                writer.writeheader()
                writer.writerows(self.history)
        return path

    def save_checkpoint(self) -> Path:
        self.artifacts_dir.mkdir(parents=True, exist_ok=True)
        path = self.artifacts_dir / f"{self.run_name}_best.pt"
        torch.save(
            {"model_state": self.model.state_dict(), "cfg": self.cfg, "history": self.history},
            path,
        )
        return path
