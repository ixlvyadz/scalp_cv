import pandas as pd

df = pd.read_csv("reports/leakage.csv")
HAM_MAX = 4   # set after reviewing the sheets: keep pairs at or below this distance out of val/test

bad = df[df.hamming <= HAM_MAX]
for split in ["val", "test"]:
    paths = sorted(set(bad[bad.split == split].image))
    with open(f"reports/exclude_{split}.txt", "w") as f:
        f.write("\n".join(paths))
    print(split, len(paths), "images excluded")

print("\ncross-class pairs to inspect by hand:")
print(df[df.cross_class][["image", "match_image", "hamming"]].to_string())