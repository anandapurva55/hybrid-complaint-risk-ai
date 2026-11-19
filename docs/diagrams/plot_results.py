"""Result figures for the README, drawn from evaluation_results.json and the trained classical model.

    python docs/diagrams/plot_results.py

Colour follows the entity: orange = the "high" tier, blue = the "low" tier / the current system,
grey = the original version. Blue/orange pair validated with the dataviz skill's validator.
"""
import json
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "docs" / "images"
BLUE, ORANGE, GREY = "#2a78d6", "#eb6834", "#9a9992"   # blue = low tier / current system, orange = high tier
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e6e5e1"
CLASSES = ["low", "medium", "high"]
plt.rcParams.update({"font.family": "DejaVu Sans", "text.color": INK, "axes.labelcolor": MUTED,
                     "xtick.color": MUTED, "ytick.color": MUTED, "axes.edgecolor": GRID})


def style(ax):
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.tick_params(length=0)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def overview():
    ev = json.load(open(ROOT / "results" / "evaluation_results.json"))
    old = ev["historical_reference_old_hybrid"]
    new = ev["system"]
    metrics = ["Recall on high", "Precision on high", "Macro-F1"]
    v_old = [old["high_recall"], old["high_precision"], old["macro_f1"]]
    v_new = [new["per_class"]["high"]["recall"], new["per_class"]["high"]["precision"], new["macro_f1"]]

    fig, (a, b) = plt.subplots(1, 2, figsize=(11.5, 4.4), gridspec_kw={"width_ratios": [1.15, 1]}, facecolor="white")
    x, w = np.arange(3), 0.36
    style(a)
    r1 = a.bar(x - w / 2 - 0.01, v_old, w, color=GREY, label="Original version (constant CNN term)")
    r2 = a.bar(x + w / 2 + 0.01, v_new, w, color=BLUE, label="Current (DistilBERT + review rules)")
    for bars in (r1, r2):
        for r in bars:
            a.text(r.get_x() + r.get_width() / 2, r.get_height() + 0.015, f"{r.get_height():.2f}", ha="center", fontsize=10, color=INK)
    a.set_xticks(x, metrics, fontsize=10)
    a.set_ylim(0, 1)
    a.set_title("Same test set (3,397 reviews), before and after", fontsize=11, loc="left", color=INK)
    a.legend(frameon=False, fontsize=9, loc="upper left", bbox_to_anchor=(0, 1.0))

    cm = np.array(new["confusion_matrix_rows_true_cols_pred"], dtype=float)
    norm = cm / cm.sum(axis=1, keepdims=True)
    b.imshow(norm, cmap=matplotlib.colors.LinearSegmentedColormap.from_list("seq", ["#ffffff", BLUE]), vmin=0, vmax=1)
    for i in range(3):
        for j in range(3):
            b.text(j, i, f"{norm[i, j]:.0%}\n({int(cm[i, j])})", ha="center", va="center", fontsize=10,
                   color="white" if norm[i, j] > 0.55 else INK)
    b.set_xticks(range(3), CLASSES)
    b.set_yticks(range(3), CLASSES)
    b.set_xlabel("Predicted tier")
    b.set_ylabel("True tier")
    b.set_title("Where the predictions land (rows sum to 100%)", fontsize=11, loc="left", color=INK)
    for s in b.spines.values():
        s.set_visible(False)
    b.tick_params(length=0)
    fig.tight_layout()
    fig.savefig(OUT / "results_overview.png", dpi=160, facecolor="white")
    plt.close(fig)


def top_terms(k=12):
    pipe = joblib.load(ROOT / "models" / "classical_model.pkl")
    tfidf = pipe.named_steps["preprocessor"].named_transformers_["tfidf"]
    names = np.array(tfidf.get_feature_names_out())
    coef = pipe.named_steps["clf"].coef_[:, : len(names)]        # text columns come first
    classes = list(pipe.classes_)
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.6), facecolor="white")
    for ax, cls, color, title in ((axes[0], "high", ORANGE, "Words that push a review towards HIGH"),
                                  (axes[1], "low", BLUE, "Words that push a review towards LOW")):
        c = coef[classes.index(cls)]
        idx = np.argsort(c)[-k:]
        ax.barh(range(k), c[idx], color=color, height=0.62)
        ax.set_yticks(range(k), names[idx], fontsize=10)
        ax.set_title(title, fontsize=11, loc="left", color=INK)
        ax.set_xlabel("weight in the classical model (larger = stronger push)")
        for s in ("top", "right", "bottom"):
            ax.spines[s].set_visible(False)
        ax.tick_params(length=0)
        ax.xaxis.grid(True, color=GRID, linewidth=0.8)
        ax.set_axisbelow(True)
    fig.tight_layout()
    fig.savefig(OUT / "top_terms.png", dpi=160, facecolor="white")
    plt.close(fig)


if __name__ == "__main__":
    overview()
    top_terms()
    print("wrote", OUT)
