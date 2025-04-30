import io,json, yaml, re
from pathlib import Path

text = Path("ricardo_gonzalez_fact_store.json").read_text(encoding='utf-8')
FACT = json.loads(text)
with io.open("hypothesis_templates.yml", "r", encoding="utf-8") as f:
    TPL = yaml.safe_load(f)

def walk(node, path=""):
    """Yield (field_path, key, value) leaves."""
    if isinstance(node, dict):
        for k, v in node.items():
            yield from walk(v, f"{path}.{k}" if path else k)
    else:
        yield path, path.split(".")[-1], node

def render(field, key, val):
    # wildcard resolution
    template = next((TPL[p] for p in (field, re.sub(r"\.[^.]+$", ".*", field)) if p in TPL), None)
    if not template:
        return None
    # publication splitter
    if field == "publication.title_venue_year_link":
        title, venue, year, *_ = val.split("_", 3)
        return template.format(title=title, venue=venue, year=year)
    # generic scalar/list
    return template.format(k=key.replace("_", " "), v=val)

hypotheses = []
for f, k, v in walk(FACT):
    vals = v if isinstance(v, list) else [v]
    for item in vals:
        h = render(f, k, item)
        if h:
            hypotheses.append({"field": f, "hypothesis": h})

json.dump(hypotheses, open("ricardo_hypotheses.json", "w"),
          ensure_ascii=False, indent=2)
print(f"✅ wrote {len(hypotheses)} hypotheses")
