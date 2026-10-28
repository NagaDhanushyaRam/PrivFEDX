import pandas as pd
import numpy as np
import wfdb
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from src.data.explore_metadata import ROOT
from src.data.ptbxl_meta import SUPERCLASSES, load_metadata

NOISE = ["baseline_drift", "static_noise", "burst_noise", "electrodes_problems"]
FLAGS = NOISE + ["pacemaker", "extra_beats"]# extra_beats is a rhythm annotation, kept for completeness
CLIENT_SITES = [0, 1, 2, 3]
BLUE, INK, MUTED = "#2b6cb0", "#1f2933", "#6b7280"
WAVE_OUT = Path(__file__).resolve().parents[2] / "results" / "eda" / "waveforms"
PLOT_IDS = [10846, 9574, 19833, 11989, 251, 2015]

def flag_summary(df):
    flagged = pd.DataFrame({f: df[f].notna() for f in FLAGS})   # any text in the cell = flagged
    print(f"records per flag (of {len(df):,}):")
    print(flagged.sum().to_string())
    print("records with >=1 noise flag:", int(flagged[NOISE].any(axis=1).sum()))

    labeled = df[df["superclasses"].str.len() > 0]
    rows = {f: labeled.loc[flagged.loc[labeled.index, f], SUPERCLASSES].mean() * 100 for f in FLAGS}
    rows["all labeled"] = labeled[SUPERCLASSES].mean() * 100
    print("\nclass prevalence (%) among labeled records carrying each flag:")
    print(pd.DataFrame(rows).T.round(1).assign(
        n=[int(flagged.loc[labeled.index, f].sum()) for f in FLAGS] + [len(labeled)]))
    
def pick_sample(df, per_site=4, seed=0):
    lab = df[df["superclasses"].str.len() > 0]
    picks = [lab[lab["site"] == s].sample(per_site, random_state=seed).assign(why=f"random, site {s}")
             for s in CLIENT_SITES]
    for flag, k in [("pacemaker", 2), ("burst_noise", 1), ("baseline_drift", 1), ("electrodes_problems", 1)]:
        picks.append(lab[lab[flag].notna()].sample(k, random_state=seed).assign(why=f"targeted: {flag}"))
    sample = pd.concat(picks)
    sample = sample[~sample.index.duplicated()]  # a targeted pick may coincide with a random one
    sample["flags"] = sample[FLAGS].notna().apply(lambda r: ",".join(r.index[r]), axis=1)
    return sample[["site", "strat_fold", "why", "flags", "filename_lr"] + SUPERCLASSES]

def summarize_record(filename_lr):
    rec = wfdb.rdrecord(str(ROOT / filename_lr))   # filename_lr has no extension
    x = rec.p_signal                                # physical units (mV), not the raw ADC ints
    std = x.std(axis=0)                             # per-lead std over time
    return {
        "shape": x.shape, "dtype": str(x.dtype), "fs": rec.fs, "units": "/".join(sorted(set(rec.units))),
        "nan": int(np.isnan(x).sum()),
        "flat": int((std == 0).sum()), "min_std": round(float(std.min()), 4),
        "abs_max": round(float(np.abs(x).max()), 2),
        "ptp_med": round(float(np.median(np.ptp(x, axis=0))), 2),   # median-over-leads peak-to-peak
        "leads": ",".join(rec.sig_name),
    }

def plot_record(df, ecg_id):
    row = df.loc[ecg_id]
    rec = wfdb.rdrecord(str(ROOT / row["filename_lr"]))
    x, t = rec.p_signal, np.arange(rec.sig_len) / rec.fs
    lim = np.abs(x).max() * 1.05                      # one shared y-range so leads are comparable

    fig, axes = plt.subplots(12, 1, figsize=(10, 9), sharex=True, sharey=True)
    for i, ax in enumerate(axes):
        ax.plot(t, x[:, i], color=BLUE, linewidth=0.8)
        ax.set_ylabel(rec.sig_name[i], rotation=0, ha="right", va="center", fontsize=9)
        ax.set_yticks([])
        for side in ("top", "right", "left"):
            ax.spines[side].set_visible(False)
        if i < 11:
            ax.spines["bottom"].set_visible(False)
            ax.tick_params(bottom=False)
    axes[0].set_ylim(-lim, lim)                       # sharey=True applies it to every lead
    axes[-1].set_xlabel("Time (s)")

    labels = ",".join(c for c in SUPERCLASSES if row[c])
    flags = "; ".join(f"{c}={row[c]!r}" for c in FLAGS if pd.notna(row[c]))
    fig.suptitle(f"ecg_id {ecg_id} | site {row['site']:.0f} | labels: {labels}",
                 x=0.01, ha="left", color=INK, fontsize=11)
    fig.text(0.01, 0.005, f"flags: {flags or 'none'}   |   every lead drawn at the same scale, +/-{lim:.2f} mV",
             fontsize=8.5, color=MUTED)
    fig.tight_layout(rect=(0, 0.02, 1, 0.97))
    fig.savefig(WAVE_OUT / f"ecg_{ecg_id}.png", dpi=150)
    plt.close(fig)
    
def main():
    df = load_metadata(ROOT)
    flag_summary(df)
    sample = pick_sample(df)
    print("\n", sample.to_string())
    summary = pd.DataFrame([summarize_record(f) for f in sample["filename_lr"]], index=sample.index)
    print("distinct lead orders:", summary["leads"].unique())
    out = sample[["site", "why"]].join(summary.drop(columns="leads"))
    print(out.to_string())
    print(out.groupby("site")[["abs_max", "ptp_med"]].agg(["min", "median", "max"]).round(2))
    WAVE_OUT.mkdir(parents=True, exist_ok=True)
    for ecg_id in PLOT_IDS:
        plot_record(df, ecg_id)
    print("wrote waveform plots to", WAVE_OUT)

if __name__ == "__main__":
    main()