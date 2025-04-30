#!/usr/bin/env python3
"""
evaluate.py  –  score model answers against a single fact/hypothesis set

Usage
-----
python evaluate.py run_results.csv \
    --facts fact_store/ricardo_gonzalez_fact_store.json \
    --hypos hypotheses/ricardo_hypotheses.json \
    --prompts prompts.yml
"""
import openai
import json, yaml, re, argparse, pandas as pd
from pathlib import Path
from openai import OpenAI

# tiny MNLI model
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch, torch.nn.functional as F

def load_api_keys():
    """Load API keys from the api_keys.yml file."""
    api_keys_path = Path(__file__).parent / "configs" / "api_keys.yml"
    if not api_keys_path.exists():
        raise FileNotFoundError(
            "api_keys.yml not found. Please add api_keys.yml to configs "
            "and fill in your API keys."
        )
    with open(api_keys_path) as f:
        return yaml.safe_load(f)
# ─────────────────────────────────────────────────────────────────────
# 0.  CLI  •  LOAD RUN CSV
# ---------------------------------------------------------------------
def load_run_csv(path: str) -> list[dict]:
    df = pd.read_csv(path, encoding="utf-8")
    must = {"prompt_id", "response"}
    if not must.issubset(df.columns):
        raise ValueError(f"CSV must contain {', '.join(must)} columns")
    return df.to_dict(orient="records")


# ─────────────────────────────────────────────────────────────────────
# 1.  LOAD GOLD DATA (no loops, single filenames)
# ---------------------------------------------------------------------
def load_facts(path: str) -> dict:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def load_hypos(path: str) -> dict[str, list[str]]:
    by_field: dict[str, list[str]] = {}
    with open(path, encoding="utf-8") as f:
        for entry in json.load(f):
            by_field.setdefault(entry["field"], []).append(entry["hypothesis"])
    return by_field


def load_prompts(path: str) -> dict[str, dict]:
    with open(path, encoding="utf-8") as f:
        plist = yaml.safe_load(f)
    return {p["id"]: p for p in plist}


# ─────────────────────────────────────────────────────────────────────
# 2.  AUTO-FAIL REGEX  (cheap contradiction filter)
# ---------------------------------------------------------------------
NEG     = re.compile(r"\b(no|not|never|formerly|previously|ex-|until)\b", re.I)
KEY_ENT = re.compile(r"\bcornell (university|tech)\b", re.I)  # adjust as needed


def auto_fail_regex(answer: str) -> bool:
    a = answer.lower()
    return bool(NEG.search(a) and KEY_ENT.search(a))


# ─────────────────────────────────────────────────────────────────────
# 3.  MINI NLI MODEL  (fast entailment)
# ---------------------------------------------------------------------
_tok = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
_nli = AutoModelForSequenceClassification.from_pretrained(
    "MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli"
).eval()


def entails(premise: str, hypothesis: str, thr: float = 0.80) -> bool:
    pair = _tok(premise, hypothesis, return_tensors="pt", truncation=True)
    with torch.no_grad():
        probs = F.softmax(_nli(**pair).logits, dim=-1)
    return probs[0, 2].item() >= thr  # index 2 = entailment


# ─────────────────────────────────────────────────────────────────────
# 4.  GPT-4o JUDGE (borderline cases only)
# ---------------------------------------------------------------------
def llm_judge(answer: str, hypotheses: list[str], question: str) -> str:
    prompt = f"""
You are grading an answer to the question:
\"{question}\"

Ground-truth claims (all must be correct):
{json.dumps(hypotheses, ensure_ascii=False, indent=2)}

Model’s answer:
\"\"\"{answer}\"\"\"

Respond with exactly one word:
PASS      – all claims supported
FAIL      – a claim contradicted
UNCERTAIN – evidence unclear
""".strip()

    # ── new OpenAI client style ───────────────────────────────
    API_KEYS = load_api_keys()
    openai.api_key = API_KEYS["openai"]["api_key"]
    client   = OpenAI()

    params = {
        "model": "gpt-4o-mini",
        "input": [{"role": "user", "content": prompt}],
        "temperature": 0,
        "max_output_tokens": 16  # plenty for a one-word answer
    }

    response = client.responses.create(**params)
    return response.output_text.strip().lower()


# ─────────────────────────────────────────────────────────────────────
# 5.  EVALUATE ONE ANSWER
# ---------------------------------------------------------------------
def evaluate(prompt_id: str, answer: str,
             prompts: dict, hypos_by_field: dict) -> str:
    fields = prompts[prompt_id]["fields"]
    hyps   = [h for f in fields for h in hypos_by_field.get(f, [])]
    question = prompts[prompt_id]["question"]

    if auto_fail_regex(answer):
        return "fail"
    if all(entails(answer, h) for h in hyps):
        return "pass"
    return llm_judge(answer, hyps, question)


# ─────────────────────────────────────────────────────────────────────
# 6.  MAIN (batch score)
# ---------------------------------------------------------------------
def main() -> None:
    cli = argparse.ArgumentParser(description="Score model answers CSV")
    cli.add_argument("--answers", help="CSV with model answers")
    cli.add_argument("--facts",   required=True, help="facts JSON path")
    cli.add_argument("--hypos",   required=True, help="hypotheses JSON path")
    cli.add_argument("--prompts", required=True, help="prompts.yml path")
    cli.add_argument("--out",     help="output CSV (default *_scored.csv)")
    args = cli.parse_args()

    rows   = load_run_csv(args.answers)
    facts  = load_facts(args.facts)           # not used yet but loaded for completeness
    hypos  = load_hypos(args.hypos)
    prompts= load_prompts(args.prompts)

    for row in rows[1:]:
        print(row)
        row["score"] = evaluate(row["prompt_id"], row["response"], prompts, hypos)

    out_path = args.out or Path(args.answers).with_stem(Path(args.answers).stem + "_scored")
    pd.DataFrame(rows).to_csv(out_path, index=False, encoding="utf-8")
    print(f"✅  scores written to {out_path}")


if __name__ == "__main__":
    main()
