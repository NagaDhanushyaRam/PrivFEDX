"""Five-client partitions: sites 0-3 are clients; sites 4-50 plus no-site records form 'rest'."""
import pandas as pd

CLIENTS = ["site0", "site1", "site2", "site3", "rest"]


def assign(meta):
    """Labeled records only, with `client` and `split` columns added."""
    df = meta[meta["superclasses"].str.len() > 0].copy()
    df["client"] = df["site"].map({0.0: "site0", 1.0: "site1", 2.0: "site2", 3.0: "site3"})
    df["client"] = df["client"].fillna("rest")        # sites 4-50 AND missing site land here
    df["split"] = pd.cut(df["strat_fold"], [0, 8, 9, 10], labels=["train", "val", "test"]).astype(str)
    return df