"""EDA figures for PTB-XL metadata -> results/eda/."""
from pathlib import Path

import matplotlib
matplotlib.use("Agg") 
import matplotlib.pyplot as plt
import pandas as pd

from src.data.explore_metadata import ROOT
from src.data.ptbxl_meta import SUPERCLASSES, load_metadata
from matplotlib.patches import Patch 

OUT = Path(__file__).resolve().parents[2] / "results" / "eda"
BLUE, INK, MUTED, GRID = "#2b6cb0", "#1f2933", "#6b7280", "#e5e7eb"
LIGHT = "#9dbbe0"
CLIENT_SITES = [0, 1, 2, 3]
CLIENT_ORDER = ["Site 0", "Site 1", "Site 2", "Site 3", "Rest"]
VMAX = 50
SMALL_SITE_MIN = 50


def load_all():
    return load_metadata(ROOT)


def fig_superclass_counts(labeled):
    counts = labeled[SUPERCLASSES].sum().sort_values()  # ascending: barh draws first item at the bottom
    pct = counts / len(labeled) * 100

    fig, ax = plt.subplots(figsize=(7, 3.6))
    ax.barh(counts.index, counts.values, color=BLUE, height=0.65)
    for y, (c, p) in enumerate(zip(counts.values, pct.values)):
        ax.text(c + counts.max() * 0.01, y, f"{c:,}  ({p:.1f}%)", va="center", color=INK, fontsize=10)
    ax.set_xlim(0, counts.max() * 1.25)  # headroom so the NORM label isn't clipped
    ax.set_xlabel("Records carrying the label")
    ax.set_title(f"PTB-XL superclasses (n = {len(labeled):,} labeled records)", loc="left", color=INK)
    ax.grid(axis="x", color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.tick_params(axis="y", length=0)
    fig.text(0.01, 0.01, "Multi-label: one record can carry several classes, so % do not sum to 100.",
             fontsize=8.5, color=MUTED)
    fig.tight_layout(rect=(0, 0.05, 1, 1))  
    fig.savefig(OUT / "a_superclass_counts.png", dpi=200)
    plt.close(fig)

def fig_site_sizes(df):
    sizes = df["site"].value_counts()          # descending; NaN dropped, so no-site records are not counted
    sizes.index = sizes.index.astype(int)      # 0.0 -> 0 for tick labels
    n_missing = int(df["site"].isna().sum())
    n = len(sizes)
    colors = [BLUE if s in CLIENT_SITES else LIGHT for s in sizes.index]

    fig, ax = plt.subplots(figsize=(10, 4.2))
    ax.bar(range(n), sizes.values, color=colors, width=0.75)
    ax.set_yscale("log")
    ax.set_ylim(1, sizes.max() * 8)            # headroom for the rotated value labels
    ax.axhline(SMALL_SITE_MIN, color=MUTED, linestyle="--", linewidth=1)
    ax.text(n - 0.5, SMALL_SITE_MIN * 1.15, f"{SMALL_SITE_MIN} records", ha="right", color=MUTED, fontsize=9)
    for i, v in enumerate(sizes.values[:6]):   # label the 6 largest bars
        ax.text(i, v * 1.15, f"{v:,}", ha="center", va="bottom", rotation=90, fontsize=9, color=INK)

    ax.set_xticks(range(n))
    ax.set_xticklabels(sizes.index, rotation=90, fontsize=7)
    ax.set_xlim(-0.7, n - 0.3)
    ax.set_xlabel("Site ID, ranked by number of records")
    ax.set_ylabel("Records (log scale)")
    ax.set_title(f"Records per site ({n} sites, all {len(df):,} records)", loc="left", color=INK)
    ax.grid(axis="y", color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.legend(handles=[Patch(color=BLUE, label="Sites 0-3: one client each"),
                       Patch(color=LIGHT, label="Sites 4-50: pooled into the 'rest' client")],
              frameon=False, loc="upper right", fontsize=9)
    fig.text(0.01, 0.01, f"{n_missing} records with no site ID are not shown.", fontsize=8.5, color=MUTED)
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    fig.savefig(OUT / "b_site_sizes.png", dpi=200)
    plt.close(fig)
    
def fig_prevalence_by_site(labeled):
    site = labeled["site"]
    groups = {f"Site {s}": labeled[site == s] for s in CLIENT_SITES}
    groups["Rest"] = labeled[~site.isin(CLIENT_SITES)]  
    assert sum(len(g) for g in groups.values()) == len(labeled)  # the 5 clients partition the labeled set
    groups["Overall"] = labeled

    table = pd.DataFrame({name: g[SUPERCLASSES].mean() * 100 for name, g in groups.items()}).T
    ns = [len(g) for g in groups.values()]

    fig, ax = plt.subplots(figsize=(7, 4.2))
    ax.imshow(table.values, cmap="Blues", vmin=0, vmax=VMAX, aspect="auto")
    for i in range(table.shape[0]):
        for j in range(table.shape[1]):
            v = table.values[i, j]
            ax.text(j, i, f"{v:.1f}", ha="center", va="center", fontsize=11,
                    color="white" if v > 0.75 * VMAX else INK)
    ax.set_xticks(range(len(SUPERCLASSES)))
    ax.set_xticklabels(SUPERCLASSES)
    ax.set_yticks(range(len(ns)))
    ax.set_yticklabels([f"{name}  (n={n:,})" for name, n in zip(table.index, ns)])
    ax.xaxis.tick_top()
    ax.tick_params(length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.axhline(4.5, color="white", linewidth=5)  # visual gap before the Overall row
    ax.set_title("Class prevalence (%) by client", loc="left", color=INK, pad=28)
    fig.text(0.01, 0.01, "% of labeled records carrying each class; rows do not sum to 100 (multi-label).\n"
                         "Rest = sites 4-50 plus records with no site ID.", fontsize=8.5, color=MUTED)
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    fig.savefig(OUT / "c_prevalence_by_site.png", dpi=200)
    plt.close(fig)

def fig_site_share_by_fold(labeled):
    client = labeled["site"].map(lambda s: f"Site {int(s)}" if s in CLIENT_SITES else "Rest")
    counts = pd.crosstab(client, labeled["strat_fold"]).reindex(CLIENT_ORDER)  # crosstab sorts rows alphabetically
    print(counts)
    share = counts / counts.sum() * 100                                          # each fold column sums to 100
    assert ((share.sum() - 100).abs() < 1e-9).all()

    fig, ax = plt.subplots(figsize=(8, 3.8))
    ax.imshow(share.values, cmap="Blues", vmin=0, vmax=VMAX, aspect="auto")
    for i in range(share.shape[0]):
        for j in range(share.shape[1]):
            v = share.values[i, j]
            ax.text(j, i, f"{v:.1f}", ha="center", va="center", fontsize=9,
                    color="white" if v > 0.75 * VMAX else INK)
    xt = [str(c) for c in share.columns]
    xt[-2:] = [xt[-2] + "\nval", xt[-1] + "\ntest"]
    ax.set_xticks(range(share.shape[1]))
    ax.set_xticklabels(xt)
    ax.set_yticks(range(share.shape[0]))
    ax.set_yticklabels(share.index)
    ax.tick_params(length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.axvline(7.5, color="white", linewidth=5)  # folds 1-8 (train) | 9-10 (val, test)
    ax.set_xlabel("strat_fold")
    ax.set_title("Client share (%) of each strat_fold", loc="left", color=INK)
    fig.text(0.01, 0.01, "Each column sums to 100%. Folds 1-8 train, 9 val, 10 test.", fontsize=8.5, color=MUTED)
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    fig.savefig(OUT / "d_site_share_by_fold.png", dpi=200)
    plt.close(fig)
    
def main():
    OUT.mkdir(parents=True, exist_ok=True)
    df = load_all()
    labeled = df[df["superclasses"].str.len() > 0]
    fig_superclass_counts(labeled)
    fig_site_sizes(df)
    fig_prevalence_by_site(labeled)
    fig_site_share_by_fold(labeled)
    print("wrote figures to", OUT)


if __name__ == "__main__":
    main()