#!/usr/bin/env python3
"""Extract epoch-level MLM losses from a Lightning console log and plot them."""

from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt


EPOCH_RE = re.compile(r"Epoch (?P<epoch>\d+):")
TRAIN_RE = re.compile(r"train/loss=(?P<value>\d+(?:\.\d+)?)")
VAL_RE = re.compile(r"val/loss=(?P<value>\d+(?:\.\d+)?)")
TRAIN_PPL_RE = re.compile(r"train/perplexity=(?P<value>\d+(?:\.\d+)?)")
VAL_PPL_RE = re.compile(r"val/perplexity=(?P<value>\d+(?:\.\d+)?)")
ELAPSED_RE = re.compile(r"\[(?P<elapsed>\d+:\d{2}:\d{2})<")
CHECKPOINT_RE = re.compile(
    r"Epoch (?P<epoch>\d+), global step (?P<step>\d+): "
    r"'val/loss' reached (?P<value>\d+(?:\.\d+)?)"
)


def console_records(path: Path, chunk_size: int = 8 * 1024 * 1024):
    """Yield terminal refresh records split on either CR or LF."""

    pending = b""
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            pending += chunk
            records = re.split(br"[\r\n]+", pending)
            pending = records.pop()
            for record in records:
                if record:
                    yield record.decode("utf-8", errors="replace")
    if pending:
        yield pending.decode("utf-8", errors="replace")


def extract_epoch_metrics(path: Path) -> list[dict[str, object]]:
    latest: dict[int, dict[str, object]] = {}
    checkpoints: dict[int, tuple[int, float]] = {}

    for record in console_records(path):
        checkpoint = CHECKPOINT_RE.search(record)
        if checkpoint:
            checkpoints[int(checkpoint.group("epoch"))] = (
                int(checkpoint.group("step")),
                float(checkpoint.group("value")),
            )

        epoch_match = EPOCH_RE.search(record)
        if not epoch_match:
            continue
        epoch = int(epoch_match.group("epoch"))
        row = latest.setdefault(epoch, {"log_epoch": epoch})

        for key, pattern in (
            ("train_loss", TRAIN_RE),
            ("displayed_val_loss", VAL_RE),
            ("train_perplexity", TRAIN_PPL_RE),
            ("val_perplexity", VAL_PPL_RE),
        ):
            matches = list(pattern.finditer(record))
            if matches:
                row[key] = float(matches[-1].group("value"))

        elapsed = ELAPSED_RE.search(record)
        if elapsed:
            row["elapsed"] = elapsed.group("elapsed")

    rows: list[dict[str, object]] = []
    for epoch, (step, exact_val_loss) in sorted(checkpoints.items()):
        row = dict(latest.get(epoch, {"log_epoch": epoch}))
        if "train_loss" not in row:
            raise RuntimeError(f"No epoch-level train/loss found for epoch {epoch}")
        row["completed_epoch"] = epoch + 1
        row["global_step"] = step
        row["val_loss"] = exact_val_loss
        rows.append(row)

    if not rows:
        raise RuntimeError("No completed epoch checkpoint records were found")
    return rows


def write_source_data(rows: list[dict[str, object]], path: Path) -> None:
    fields = [
        "completed_epoch",
        "log_epoch",
        "global_step",
        "train_loss",
        "val_loss",
        "train_perplexity",
        "val_perplexity",
        "elapsed",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def configure_style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
            "font.size": 8,
            "axes.labelsize": 9,
            "axes.titlesize": 11,
            "axes.linewidth": 0.8,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "legend.frameon": False,
            "pdf.fonttype": 42,
            "svg.fonttype": "none",
        }
    )


def plot(rows: list[dict[str, object]], output_stem: Path) -> None:
    configure_style()
    epochs = [int(row["completed_epoch"]) for row in rows]
    train = [float(row["train_loss"]) for row in rows]
    val = [float(row["val_loss"]) for row in rows]
    steps = [int(row["global_step"]) for row in rows]

    train_color = "#2F6FB0"
    val_color = "#D97917"
    fig, ax = plt.subplots(figsize=(6.8, 4.15), constrained_layout=True)

    ax.plot(
        epochs,
        train,
        color=train_color,
        marker="o",
        markersize=6,
        linewidth=2.1,
        label="Training loss",
        zorder=3,
    )
    ax.plot(
        epochs,
        val,
        color=val_color,
        marker="s",
        markersize=5.8,
        linewidth=2.1,
        label="Validation loss",
        zorder=3,
    )

    for x, y in zip(epochs, train):
        ax.annotate(
            f"{y:.3f}",
            (x, y),
            xytext=(0, 8),
            textcoords="offset points",
            ha="center",
            color=train_color,
            fontsize=8,
            fontweight="bold",
        )
    for x, y in zip(epochs, val):
        ax.annotate(
            f"{y:.3f}",
            (x, y),
            xytext=(0, -14),
            textcoords="offset points",
            ha="center",
            color=val_color,
            fontsize=8,
            fontweight="bold",
        )

    ax.scatter(
        [epochs[-1]],
        [val[-1]],
        s=120,
        facecolors="none",
        edgecolors=val_color,
        linewidths=1.2,
        zorder=2,
    )
    ax.annotate(
        "Best checkpoint",
        (epochs[-1], val[-1]),
        xytext=(-62, -31),
        textcoords="offset points",
        arrowprops={"arrowstyle": "-", "color": "#666666", "lw": 0.8},
        color="#555555",
        fontsize=7.5,
    )

    train_drop = train[0] - train[-1]
    val_drop = val[0] - val[-1]
    summary = (
        f"Training: −{train_drop:.3f} ({100 * train_drop / train[0]:.1f}%)\n"
        f"Validation: −{val_drop:.3f} ({100 * val_drop / val[0]:.1f}%)"
    )
    ax.text(
        0.035,
        0.075,
        summary,
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=8,
        linespacing=1.5,
        bbox={"boxstyle": "round,pad=0.45", "fc": "#F5F6F7", "ec": "#D8DADC", "lw": 0.7},
    )

    ax.set_title("RNA-Mamba epoch-end pretraining loss", loc="left", pad=14, fontweight="bold")
    ax.text(
        0,
        1.025,
        f"Epoch 1 point follows {steps[0]:,} optimizer steps · final step {steps[-1]:,}",
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=8,
        color="#555555",
    )
    ax.set_xlabel("Completed epoch")
    ax.set_ylabel("Masked-language-model loss")
    ax.set_xticks(epochs)
    ax.set_xlim(min(epochs) - 0.15, max(epochs) + 0.15)
    lower = min(train + val) - 0.018
    upper = max(train + val) + 0.025
    ax.set_ylim(lower, upper)
    ax.grid(axis="y", color="#D9D9D9", linewidth=0.7, alpha=0.75)
    ax.tick_params(direction="out", length=3, width=0.8)
    ax.legend(loc="upper right", handlelength=2.4)

    output_stem.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_stem.with_suffix(".png"), dpi=400, bbox_inches="tight", facecolor="white")
    fig.savefig(output_stem.with_suffix(".pdf"), bbox_inches="tight", facecolor="white")
    fig.savefig(output_stem.with_suffix(".svg"), bbox_inches="tight", facecolor="white")
    fig.savefig(output_stem.with_suffix(".tiff"), dpi=600, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--log", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    rows = extract_epoch_metrics(args.log)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    source_path = args.output_dir / "pretraining_loss_source_data.csv"
    output_stem = args.output_dir / "rna_mamba_5m_pretraining_loss"
    write_source_data(rows, source_path)
    plot(rows, output_stem)

    print(f"epochs: {len(rows)}")
    print(f"train loss: {rows[0]['train_loss']:.3f} -> {rows[-1]['train_loss']:.3f}")
    print(f"val loss: {rows[0]['val_loss']:.5f} -> {rows[-1]['val_loss']:.5f}")
    print(f"source data: {source_path}")
    print(f"figure stem: {output_stem}")


if __name__ == "__main__":
    main()
