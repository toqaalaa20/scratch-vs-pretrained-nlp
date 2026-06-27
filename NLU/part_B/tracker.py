from dataclasses import dataclass, field, asdict
import csv
import json
import os
from typing import Optional
import matplotlib.pyplot as plt
from datetime import datetime


@dataclass
class EpochLog:
    epoch: int
    train_loss: float
    dev_loss: float
    dev_slot_f1: Optional[float] = None
    dev_intent_acc: Optional[float] = None


@dataclass
class Experiment:
    name: str
    model_type: str             # "gpt2" or "bert"
    checkpoint: str             # e.g. "openai-community/gpt2", "bert-base-uncased"
    learning_rate: float
    dropout: float      = 0.1
    test_slot_f1: Optional[float]    = None
    test_slot_f1_std: Optional[float]     = None
    test_intent_acc: Optional[float] = None
    test_intent_acc_std: Optional[float]  = None
    dev_slot_f1: Optional[float]     = None
    dev_intent_acc: Optional[float]  = None
    notes: str         = ""
    timestamp: str     = field(default_factory=lambda: datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    epoch_logs: list   = field(default_factory=list)

    @property
    def score(self) -> Optional[float]:
        if self.test_slot_f1 is None or self.test_intent_acc is None:
            return None
        return (self.test_slot_f1 + self.test_intent_acc) / 2

    @property
    def status(self) -> str:
        return "done" if self.score is not None else "—"


class ExperimentTracker:
    def __init__(self, save_dir: str = "NLU/part_B/results"):
        self.save_dir = save_dir
        self.experiments: list[Experiment] = []
        os.makedirs(save_dir, exist_ok=True)
        self._load()

    # -- logging ---------------------------------------------------------

    def log(
        self,
        name: str,
        model_type: str,
        checkpoint: str,
        learning_rate: float,
        test_slot_f1: float,
        test_intent_acc: float,
        test_slot_f1_std: float    = None,
        test_intent_acc_std: float = None,
        dev_slot_f1: float    = None,
        dev_intent_acc: float = None,
        # training curves -- pass the lists your loop already built
        sampled_epochs: list   = None,
        losses_train: list     = None,
        losses_dev: list       = None,
        dev_slot_f1s: list     = None,
        dev_intent_accs: list  = None,
        dropout: float  = 0.1,
        notes: str      = "",
    ) -> Experiment:
        exp = Experiment(
            name=name, model_type=model_type, checkpoint=checkpoint,
            learning_rate=learning_rate, dropout=dropout,
            test_slot_f1=test_slot_f1, test_slot_f1_std=test_slot_f1_std,
            test_intent_acc=test_intent_acc, test_intent_acc_std=test_intent_acc_std,
            dev_slot_f1=dev_slot_f1, dev_intent_acc=dev_intent_acc, notes=notes,
        )

        if sampled_epochs and losses_train and losses_dev:
            for i, epoch in enumerate(sampled_epochs):
                exp.epoch_logs.append(EpochLog(
                    epoch=epoch,
                    train_loss=losses_train[i] if i < len(losses_train) else float("nan"),
                    dev_loss=losses_dev[i]     if i < len(losses_dev)   else float("nan"),
                    dev_slot_f1=dev_slot_f1s[i]    if dev_slot_f1s and i < len(dev_slot_f1s)    else None,
                    dev_intent_acc=dev_intent_accs[i] if dev_intent_accs and i < len(dev_intent_accs) else None,
                ))

        self.experiments.append(exp)
        self._save()
        dev_f1_str  = f"{dev_slot_f1:.3f}"    if dev_slot_f1    is not None else "—"
        dev_acc_str = f"{dev_intent_acc:.3f}" if dev_intent_acc is not None else "—"
        f1_str  = f"{test_slot_f1:.3f}"    + (f" +- {test_slot_f1_std:.3f}"    if test_slot_f1_std    is not None else "")
        acc_str = f"{test_intent_acc:.3f}" + (f" +- {test_intent_acc_std:.3f}" if test_intent_acc_std is not None else "")
        print(f"[Tracker] '{name}'  model={model_type} ({checkpoint})  "
              f"test slot F1={f1_str}  test intent acc={acc_str}  "
              f"dev slot F1={dev_f1_str}  dev intent acc={dev_acc_str}")
        return exp

    # -- persistence ------------------------------------------------------

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
            exp  = Experiment(**d)
            exp.epoch_logs = logs
            self.experiments.append(exp)
        print(f"[Tracker] Loaded {len(self.experiments)} experiment(s) from {self._path()}")

    def export_csv(self):
        path = os.path.join(self.save_dir, "experiments.csv")
        fields = ["name", "model_type", "checkpoint", "status", "learning_rate", "dropout",
                  "test_slot_f1", "test_slot_f1_std",
                  "test_intent_acc", "test_intent_acc_std",
                  "dev_slot_f1", "dev_intent_acc", "notes", "timestamp"]
        with open(path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fields)
            w.writeheader()
            for exp in self.experiments:
                w.writerow({k: getattr(exp, k, None) for k in fields})
        print(f"[Tracker] CSV -> {path}")

    # -- reporting ---------------------------------------------------------

    def summary(self):
        if not self.experiments:
            print("[Tracker] No experiments logged yet.")
            return
        col = "{:<3} {:<28} {:<6} {:<22} {:<9} {:<8} {:<16} {:<16} {:<6}"
        header = col.format("#", "Name", "Model", "Checkpoint", "LR", "Dropout",
                             "Slot F1", "Int Acc", "Status")
        sep = "-" * len(header)
        print(f"\n{sep}\n{'EXPERIMENT SUMMARY':^{len(header)}}\n{sep}")
        print(header)
        print(sep)
        best_score = -1
        for i, exp in enumerate(self.experiments):
            f1_str  = f"{exp.test_slot_f1:.3f}"    if exp.test_slot_f1    is not None else "—"
            f1_str  += f"±{exp.test_slot_f1_std:.3f}"    if exp.test_slot_f1_std    is not None else ""
            acc_str = f"{exp.test_intent_acc:.3f}" if exp.test_intent_acc is not None else "—"
            acc_str += f"±{exp.test_intent_acc_std:.3f}" if exp.test_intent_acc_std is not None else ""
            is_best = exp.score is not None and exp.score > best_score
            if is_best:
                best_score = exp.score
            print(col.format(
                i+1, exp.name[:27], exp.model_type, exp.checkpoint[:21],
                exp.learning_rate, exp.dropout, f1_str, acc_str, exp.status,
            ) + (" <-" if is_best else ""))
        print(sep)
        best_str = f"{best_score:.3f}" if best_score >= 0 else "—"
        print(f"  Total: {len(self.experiments)}  |  Best avg(F1, Acc): {best_str}\n")

    def best(self) -> Optional[Experiment]:
        scored = [e for e in self.experiments if e.score is not None]
        return max(scored, key=lambda e: e.score) if scored else None

    # -- plots ---------------------------------------------------------------

    def plot_metrics(self):
        """Grouped bar chart of test slot F1 and intent accuracy across all logged experiments."""
        finished = [e for e in self.experiments if e.score is not None]
        if not finished:
            print("[Tracker] Nothing to plot yet."); return

        names    = [e.name for e in finished]
        f1s      = [e.test_slot_f1 for e in finished]
        accs     = [e.test_intent_acc for e in finished]
        f1_stds  = [e.test_slot_f1_std or 0 for e in finished]
        acc_stds = [e.test_intent_acc_std or 0 for e in finished]
        x = range(len(names))

        fig, ax = plt.subplots(figsize=(max(7, len(names) * 1.5), 4))
        bars1 = ax.bar([i - 0.2 for i in x], f1s, width=0.4, yerr=f1_stds, capsize=3, label="Slot F1", color="#185FA5", zorder=2)
        bars2 = ax.bar([i + 0.2 for i in x], accs, width=0.4, yerr=acc_stds, capsize=3, label="Intent Acc", color="#D85A30", zorder=2)
        ax.bar_label(bars1, fmt="%.2f", padding=3, fontsize=8)
        ax.bar_label(bars2, fmt="%.2f", padding=3, fontsize=8)
        ax.set_xticks(list(x)); ax.set_xticklabels(names, rotation=20, ha="right", fontsize=9)
        ax.set_ylabel("Score"); ax.set_title("Test metrics per experiment")
        ax.legend(fontsize=9); ax.grid(axis="y", alpha=0.3, zorder=1)
        plt.tight_layout()
        path = os.path.join(self.save_dir, "metrics_comparison.png")
        plt.savefig(path, dpi=150); plt.show()
        print(f"[Tracker] Saved -> {path}")

    def plot_curves(self, exp: Experiment):
        """Two-panel: train/dev loss (left) and dev slot F1 / intent acc (right)."""
        if not exp.epoch_logs:
            print(f"[Tracker] No epoch logs for '{exp.name}'."); return

        epochs  = [e.epoch      for e in exp.epoch_logs]
        train_l = [e.train_loss for e in exp.epoch_logs]
        dev_l   = [e.dev_loss   for e in exp.epoch_logs]

        logs_with_metrics = [e for e in exp.epoch_logs if e.dev_slot_f1 is not None and e.dev_intent_acc is not None]

        fig, axes = plt.subplots(1, 2 if logs_with_metrics else 1, figsize=(13, 4))
        if not logs_with_metrics:
            axes = [axes]

        axes[0].plot(epochs, train_l, label="Train loss", color="#185FA5", linewidth=1.6)
        axes[0].plot(epochs, dev_l,   label="Dev loss",   color="#D85A30", linewidth=1.6, linestyle="--")
        axes[0].set_title("Loss"); axes[0].set_xlabel("Epoch"); axes[0].set_ylabel("Loss")
        axes[0].legend(fontsize=9); axes[0].grid(alpha=0.3)

        if logs_with_metrics:
            m_epochs = [e.epoch          for e in logs_with_metrics]
            f1s      = [e.dev_slot_f1    for e in logs_with_metrics]
            accs     = [e.dev_intent_acc for e in logs_with_metrics]
            axes[1].plot(m_epochs, f1s,  label="Dev slot F1",   color="#185FA5", linewidth=1.6, marker="o", markersize=3)
            axes[1].plot(m_epochs, accs, label="Dev intent acc", color="#D85A30", linewidth=1.6, linestyle="--", marker="s", markersize=3)
            axes[1].set_title("Dev metrics"); axes[1].set_xlabel("Epoch"); axes[1].set_ylabel("Score")
            axes[1].legend(fontsize=9); axes[1].grid(alpha=0.3)

        title = exp.name
        if exp.test_slot_f1 is not None:    title += f"  |  test F1 = {exp.test_slot_f1:.3f}"
        if exp.test_intent_acc is not None: title += f"  |  test acc = {exp.test_intent_acc:.3f}"
        fig.suptitle(title, fontsize=11)
        plt.tight_layout()
        safe = exp.name.replace(" ", "_").replace("/", "-")
        path = os.path.join(self.save_dir, f"curves_{safe}.png")
        plt.savefig(path, dpi=150); plt.show()
        print(f"[Tracker] Saved -> {path}")
