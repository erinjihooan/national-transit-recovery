"""
Diagnostic: inspect unique city values in your processed CSV
and flag likely near-duplicates (whitespace, casing, punctuation variants).
"""
import pandas as pd
from pathlib import Path
from difflib import SequenceMatcher

df = pd.read_csv("data/processed/transit_recovery_tidy.csv")

cities = sorted(df["city"].dropna().unique())
print(f"Total unique city values: {len(cities)}\n")

# 1. Show raw list so you can eyeball obvious junk (whitespace, weird chars)
print("=== All unique city values (repr, to expose hidden whitespace) ===")
for c in cities:
    print(repr(c))

# 2. Flag pairs that are suspiciously similar (likely the same city, split apart)
print("\n=== Likely near-duplicate pairs (similarity > 0.85) ===")
flagged = set()
for i, a in enumerate(cities):
    for b in cities[i+1:]:
        ratio = SequenceMatcher(None, a.lower().strip(), b.lower().strip()).ratio()
        if ratio > 0.85 and a != b:
            flagged.add((a, b, round(ratio, 2)))

for a, b, r in sorted(flagged, key=lambda x: -x[2]):
    print(f"  {a!r}  <->  {b!r}   (similarity={r})")

# 3. Cities with very few rows (often small/rural UZAs, or parsing artifacts)
print("\n=== Cities with fewer than 12 rows (possible one-off artifacts) ===")
counts = df["city"].value_counts()
sparse = counts[counts < 12]
print(sparse)