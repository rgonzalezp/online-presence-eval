import json, re
import pandas as pd

CSV_URL = (
    "https://docs.google.com/spreadsheets/d/1F1kWBR1eZ_X6DLz1b1cIZLhROOiDGYZjPUFdoGVvezo/export?format=csv&gid=286826835#gid=286826835"
)

def slug(text: str) -> str:
    return re.sub(r"\s+", "_", text.lower().strip())

facts_df = pd.read_csv(CSV_URL)
facts_df.columns = facts_df.columns.str.strip()      # ① trim header junk

person: dict[str, dict] = {}

for _, row in facts_df.iterrows():
    cat, attr, val = (row[c] if pd.notna(row[c]) else None          # ② keep None for blanks
                       for c in ("category", "attribute", "value"))
    if cat is None or attr is None or val is None:
        continue                                                   # skip incomplete rows

    cat_slug = slug(cat)                                           # ③ slug category too
    attr_slug = slug(attr)

    bucket = person.setdefault(cat_slug, {})
    previous = bucket.get(attr_slug)

    if previous is None:
        bucket[attr_slug] = val
    elif isinstance(previous, list):
        previous.append(val)
    else:
        bucket[attr_slug] = [previous, val]

with open("ricardo_gonzalez_fact_store.json", "w", encoding="utf-8") as f:     # ④ force UTF-8
    json.dump(person, f, ensure_ascii=False, indent=2)
