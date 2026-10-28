"""PTB-XL metadata EDA: labels, sites, demographics, folds."""
from pathlib import Path

import pandas as pd

from src.data.ptbxl_meta import SUPERCLASSES, load_metadata

ROOT = Path(__file__).resolve().parents[2] / "data" / "raw" / "ptbxl"
BIG_SITE_MIN = 500  # sites with at least this many records get their own row in the site x class table


def main():
    df = load_metadata(ROOT)
    n_labels = df["superclasses"].str.len()
    labeled = df[n_labels > 0]
    print(f"records={len(df)}  patients={df.patient_id.nunique()}  "
          f"with >=1 superclass={len(labeled)}  no superclass={(n_labels == 0).sum()}")

    print("\n== Superclass labels (multi-label) ==")
    print("positives per class:", df[SUPERCLASSES].sum().to_dict())
    print("prevalence % of labeled:", (labeled[SUPERCLASSES].mean() * 100).round(1).to_dict())
    print("labels per record:", n_labels.value_counts().sort_index().to_dict())

    print("\n== Sites ==")
    sizes = df["site"].value_counts()
    print(f"unique sites={len(sizes)}  missing site={df.site.isna().sum()}")
    print("records per site, top 6:", sizes.head(6).astype(int).to_dict())
    print(f"sites with <50 records: {(sizes < 50).sum()} "
          f"(hold {int(sizes[sizes < 50].sum())} records); median site size={sizes.median():.0f}")
    big = sizes[sizes >= BIG_SITE_MIN].index
    print(f"sites with >={BIG_SITE_MIN} records: {[int(s) for s in big]} "
          f"= {sizes[big].sum() / len(df):.1%} of records")

    print("\n== Class prevalence % by site (labeled records; big sites vs rest) ==")
    l = labeled.assign(grp=labeled["site"].where(labeled["site"].isin(big), -1))
    print((l.groupby("grp")[SUPERCLASSES].mean() * 100).round(1).assign(n=l.groupby("grp").size()))
    print("(grp -1 = all other sites incl. missing)")

    print("\n== Site x device (records) ==")
    dev = df["device"].str.strip().str.split().str[0]
    print(pd.crosstab(df["site"].where(df["site"].isin(big), -1), dev))

    print("\n== Demographics ==")
    age = df["age"].where(df["age"] < 300)  # 300 = anonymised age >89 in PTB-XL
    print(f"age==300 (masked): {(df.age == 300).sum()}; age median={age.median():.0f} "
          f"IQR=({age.quantile(.25):.0f},{age.quantile(.75):.0f}); age<18: {(age < 18).sum()}")
    print("sex counts (0/1 coding per PTB-XL docs):", df["sex"].value_counts().to_dict())
    print("height missing:", df.height.isna().sum(), " weight missing:", df.weight.isna().sum())
    by_site = df.assign(age=age, grp=df["site"].where(df["site"].isin(big), -1)).groupby("grp")
    print(by_site.agg(age_median=("age", "median"), sex_mean=("sex", "mean"), n=("sex", "size")).round(2))

    print("\n== Folds / leakage ==")
    print("patients appearing in >1 strat_fold:", int((df.groupby("patient_id").strat_fold.nunique() > 1).sum()))
    print("patients appearing at >1 site:",
          int((df.dropna(subset=["site"]).groupby("patient_id").site.nunique() > 1).sum()))
    print("share validated_by_human, folds 1-8 / 9-10:",
          round(df[df.strat_fold <= 8].validated_by_human.mean(), 2),
          round(df[df.strat_fold >= 9].validated_by_human.mean(), 2))
    grp = df["site"].where(df["site"].isin(big), -1)
    print("site share (%) per strat_fold:")
    print((pd.crosstab(grp, df["strat_fold"], normalize="columns") * 100).round(0).astype(int))


if __name__ == "__main__":
    main()
