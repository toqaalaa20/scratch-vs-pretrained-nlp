from dataclasses import dataclass, field, asdict
import csv
import json
import math
import os
from typing import Optional
import matplotlib.pyplot as plt
from datetime import datetime


@dataclass
class EpochLog:
    epoch: int
    train_loss: float
    dev_loss: float
    train_ppl: Optional[float] = None
    dev_ppl: Optional[float] = None


@dataclass
class Experiment:
    name: str
    rank: int
    alpha: float
    learning_rate: float
    test_ppl: Optional[float] = None
    dev_ppl: Optional[float] = None
    train_ppl: Optional[float] = None
    notes: str = ""
    timestamp: str = field(default_factory=lambda: datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    epoch_logs: list = field(default_factory=list)

    @property
    def status(self) -> str:
        if self.test_ppl is None:
            return "—"
        return "pass" if self.test_ppl < 250 else "fail"


class ExperimentTracker:
    def __init__(self, save_dir: str = "LM/part_B/results"):
        self.save_dir = save_dir
        self.experiments: list[Experiment] = []
        os.makedirs(save_dir, exist_ok=True)
        self._load()

    # ── logging ───────────────────────────────────────────────────────────────

    def log(
        self,
        name: str,
        rank: int,
        alpha: float,
        learning_rate: float,
        test_ppl: float,
        dev_ppl: float = None,
        train_ppl: float = None,
        # training curves — pass the lists your loop already built
        sampled_epochs: list = None,
        losses_train: list = None,
        losses_dev: list = None,
        ppls_train: list = None,
        ppls_dev: list = None,
        notes: str = "",
    ) -> Experiment:
        exp = Experiment(
            name=name, rank=rank, alpha=alpha, learning_rate=learning_rate,
            test_ppl=test_ppl, dev_ppl=dev_ppl, train_ppl=train_ppl, notes=notes,
        )

        if sampled_epochs and losses_train and losses_dev:
            for i, epoch in enumerate(sampled_epochs):
                exp.epoch_logs.append(EpochLog(
                    epoch=epoch,
                    train_loss=losses_train[i] if i < len(losses_train) else float("nan"),
                    dev_loss=losses_dev[i] if i < len(losses_dev) else float("nan"),
                    train_ppl=ppls_train[i] if ppls_train and i < len(ppls_train) else None,
                    dev_ppl=ppls_dev[i] if ppls_dev and i < len(ppls_dev) else None,
                ))

        self.experiments.append(exp)
        self._save()
        dev_ppl_str = f"{dev_ppl:.2f}" if dev_ppl else "—"
        train_ppl_str = f"{train_ppl:.2f}" if train_ppl else "—"
        print(f"[Tracker] '{name}'  rank={rank} alpha={alpha} lr={learning_rate}  test PPL={test_ppl:.2f}  dev PPL={dev_ppl_str}  train PPL={train_ppl_str}  [{exp.status}]")
        return exp

    # ── persistence ───────────────────────────────────────────────────────────

    def _path(self):
        return os.path.join(self.save_dir, "experiments.json")

    def _save(self):
        with open(self._path(), "w") as f:
            json.dump([asdict(e) for e in self.experiments], f, indent=2)

    def _load(self):
        if not os.path.exists(self._path()):
            return
        with open(self._path()) as f:
            data = json.load(f)
        for d in data:
            logs = [EpochLog(**e) for e in d.pop("epoch_logs", [])]
            exp = Experiment(**d)
            exp.epoch_logs = logs
            self.experiments.append(exp)
        print(f"[Tracker] Loaded {len(self.experiments)} experiment(s) from {self._path()}")

    def export_csv(self):
        path = os.path.join(self.save_dir, "experiments.csv")
        fields = ["name", "status", "rank", "alpha", "learning_rate",
                  "test_ppl", "dev_ppl", "train_ppl", "notes", "timestamp"]
        with open(path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fields)
            w.writeheader()
            for exp in self.experiments:
                w.writerow({k: getattr(exp, k, None) for k in fields})
        print(f"[Tracker] CSV → {path}")

    # ── reporting ─────────────────────────────────────────────────────────────

    def summary(self):
        if not self.experiments:
            print("[Tracker] No experiments logged yet.")
            return
        col = "{:<3} {:<40} {:<6} {:<7} {:<10} {:<10} {:<10} {:<6}"
        header = col.format("#", "Name", "Rank", "Alpha", "LR", "Test PPL", "Dev PPL", "Status")
        sep = "─" * len(header)
        print(f"\n{sep}\n{'EXPERIMENT SUMMARY':^{len(header)}}\n{sep}")
        print(header)
        print(sep)
        best_ppl = math.inf
        for i, exp in enumerate(self.experiments):
            ppl_str = f"{exp.test_ppl:.2f}" if exp.test_ppl is not None else "—"
            dev_ppl_str = f"{exp.dev_ppl:.2f}" if exp.dev_ppl is not None else "—"
            is_best = exp.test_ppl and exp.test_ppl < best_ppl and exp.status == "pass"
            if is_best:
                best_ppl = exp.test_ppl
            print(col.format(
                i + 1, exp.name[:39], exp.rank, exp.alpha, exp.learning_rate,
                ppl_str, dev_ppl_str, exp.status,
            ) + (" ◄" if is_best else ""))
        print(sep)
        passing = [e for e in self.experiments if e.status == "pass"]
        best_str = f"{best_ppl:.2f}" if best_ppl < math.inf else "—"
        print(f"  Total: {len(self.experiments)}  |  Passing (PPL<250): {len(passing)}  |  Best PPL: {best_str}\n")

    def best(self) -> Optional[Experiment]:
        passing = [e for e in self.experiments if e.status == "pass" and e.test_ppl is not None]
        return min(passing, key=lambda e: e.test_ppl) if passing else None

    # ── plots ─────────────────────────────────────────────────────────────────

    def plot_ppl(self):
        """Bar chart of test PPL across all logged experiments."""
        finished = [e for e in self.experiments if e.test_ppl is not None]
        if not finished:
            print("[Tracker] Nothing to plot yet."); return

        names = [e.name for e in finished]
        ppls = [e.test_ppl for e in finished]
        colors = ["#0F6E56" if p < 250 else "#A32D2D" for p in ppls]

        fig, ax = plt.subplots(figsize=(max(7, len(names) * 1.3), 4))
        bars = ax.bar(names, ppls, color=colors, width=0.55, zorder=2)
        ax.axhline(250, color="#BA7517", linewidth=1.2, linestyle="--", label="PPL = 250")
        ax.bar_label(bars, fmt="%.1f", padding=3, fontsize=9)
        ax.set_ylabel("Test PPL"); ax.set_title("Test PPL per experiment")
        ax.legend(fontsize=9); ax.grid(axis="y", alpha=0.3, zorder=1)
        plt.xticks(rotation=20, ha="right", fontsize=9)
        plt.tight_layout()
        path = os.path.join(self.save_dir, "ppl_comparison.png")
        plt.savefig(path, dpi=150); plt.show()
        print(f"[Tracker] Saved → {path}")

    def plot_curves(self, exp: Experiment):
        """Two-panel: train/dev loss (left) and train/dev PPL (right)."""
        if not exp.epoch_logs:
            print(f"[Tracker] No epoch logs for '{exp.name}'."); return

        epochs = [e.epoch for e in exp.epoch_logs]
        train_l = [e.train_loss for e in exp.epoch_logs]
        dev_l = [e.dev_loss for e in exp.epoch_logs]

        logs_with_ppl = [e for e in exp.epoch_logs if e.train_ppl is not None and e.dev_ppl is not None]

        fig, axes = plt.subplots(1, 2 if logs_with_ppl else 1, figsize=(13, 4))
        if not logs_with_ppl:
            axes = [axes]

        axes[0].plot(epochs, train_l, label="Train loss", color="#185FA5", linewidth=1.6)
        axes[0].plot(epochs, dev_l, label="Dev loss", color="#D85A30", linewidth=1.6, linestyle="--")
        axes[0].set_title("Loss"); axes[0].set_xlabel("Epoch"); axes[0].set_ylabel("Loss")
        axes[0].legend(fontsize=9); axes[0].grid(alpha=0.3)

        if logs_with_ppl:
            ppl_epochs = [e.epoch for e in logs_with_ppl]
            train_p = [e.train_ppl for e in logs_with_ppl]
            dev_p = [e.dev_ppl for e in logs_with_ppl]
            axes[1].plot(ppl_epochs, train_p, label="Train PPL", color="#185FA5", linewidth=1.6, marker="o", markersize=3)
            axes[1].plot(ppl_epochs, dev_p, label="Dev PPL", color="#D85A30", linewidth=1.6, linestyle="--", marker="s", markersize=3)
            axes[1].set_title("Perplexity"); axes[1].set_xlabel("Epoch"); axes[1].set_ylabel("PPL")
            axes[1].legend(fontsize=9); axes[1].grid(alpha=0.3)

        title = exp.name
        if exp.test_ppl: title += f"  |  test PPL = {exp.test_ppl:.2f}"
        if exp.dev_ppl: title += f"  |  dev PPL = {exp.dev_ppl:.2f}"
        fig.suptitle(title, fontsize=11)
        plt.tight_layout()
        safe = exp.name.replace(" ", "_").replace("/", "-")
        path = os.path.join(self.save_dir, f"curves_{safe}.png")
        plt.savefig(path, dpi=150); plt.show()
        print(f"[Tracker] Saved → {path}")
