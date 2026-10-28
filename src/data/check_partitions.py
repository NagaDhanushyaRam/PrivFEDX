from src.data.ptbxl_meta import load_metadata
from src.data.partitions import assign

EXPECTED = {   # (train, val, test) from the EDA doc
    "site0": (7691, 497, 483),
    "site1": (5038, 606, 598),
    "site2": (3318, 834, 867),
    "site3": (332, 115, 103),
    "rest": (705, 94, 107),
}

df = assign(load_metadata("data/raw/ptbxl"))
counts = df.groupby(["client", "split"]).size().unstack()
for client, want in EXPECTED.items():
    got = tuple(int(counts.loc[client, s]) for s in ["train", "val", "test"])
    assert got == want, (client, got, want)
assert len(df) == 21388
print(counts)
print("partitions match the doc")