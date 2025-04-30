import json, yaml, re, argparse, pandas as pd, openai
from pathlib import Path
from openai import OpenAI
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch, torch.nn.functional as F


# ───────────────────────────────────────────────────────── helpers ──
def load_api_keys():
    path = Path(__file__).parent / "configs" / "api_keys.yml"
    with open(path) as f:
        return yaml.safe_load(f)


def load_run_csv(path: str):
    df = pd.read_csv(path, encoding="utf-8")
    must = {"prompt_id", "response"}
    if not must.issubset(df.columns):
        raise ValueError("CSV must contain prompt_id and response")
    return df.to_dict(orient="records")


def load_hypos(path: str):
    by_field = {}
    with open(path, encoding="utf-8") as f:
        for e in json.load(f):
            by_field.setdefault(e["field"], []).append(e["hypothesis"])
    return by_field


def load_prompts(path: str):
    with open(path, encoding="utf-8") as f:
        plist = yaml.safe_load(f)
    return {p["id"]: p for p in plist}

# ───────────────────────────────────────────────(This model doesn't work)─ error detection ──
tokenizer_error = AutoTokenizer.from_pretrained("byviz/bylastic_classification_logs")
model_error = AutoModelForSequenceClassification.from_pretrained("byviz/bylastic_classification_logs")

def errorDetection(response)-> bool:
    inputs = tokenizer_error(response, return_tensors="pt")
    outputs = model_error(**inputs)
    probabilities = torch.nn.functional.softmax(outputs.logits, dim=-1)
    prediction = torch.argmax(probabilities, dim=-1)
    return model_error.config.id2label[prediction.item()] != "INFO"

# ────────────────────────────────────────────────────────── regex ──
NEG     = re.compile(r"\b(no|not|never|formerly|previously|ex-|until)\b", re.I)
KEY_ENT = re.compile(r"\bcornell (university|tech)\b", re.I)

def auto_fail_regex(ans: str) -> bool:
    a = ans.lower()
    return bool(NEG.search(a) and KEY_ENT.search(a))


# ────────────────────────────────────────────────────────── NLI  ──
_tok = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
_nli = AutoModelForSequenceClassification.from_pretrained(
        "MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli").eval()

def entails(premise: str, hyp: str, thr: float = .80) -> bool:
    pair = _tok(premise, hyp, return_tensors="pt", truncation=True)
    with torch.no_grad():
        probs = F.softmax(_nli(**pair).logits, dim=-1)
    return probs[0,2].item() >= thr


# ─────────────────────────────────────────────── GPT-judge prompt ──
def llm_judge(ans: str, hyps: list[str], q: str) -> tuple[str, str]:
    prompt = f"""
You are grading an answer to the question:
\"{q}\"

Ground-truth claims:
{json.dumps(hyps, ensure_ascii=False, indent=2)}

Model’s answer:
\"\"\"{ans}\"\"\"

Respond with one word: PASS / FAIL / UNCERTAIN
""".strip()

    openai.api_key = load_api_keys()["openai"]["api_key"]
    client = OpenAI()
    resp = client.responses.create(
        model="gpt-4o-mini",
        input=[{"role": "user", "content": prompt}],
        temperature=0,
        max_output_tokens=16,
    )
    return resp.output_text.strip().lower(), prompt


# ────────────────────────────────────────────── evaluate answer ──
def evaluate(prompt_id: str, ans: str, prompts: dict,
             hypos_by_field: dict, stages: set[str]) -> tuple[str, str, str]:
    """
    Returns (score, method, query)
    """
    fields = prompts[prompt_id]["fields"]
    hyps   = [h for f in fields for h in hypos_by_field.get(f, [])]
    q_text = prompts[prompt_id]["question"]

    errorDetection(ans)

    return
    # 1) auto-fail
    if "auto" in stages and auto_fail_regex(ans):
        return "fail", "auto", "NEG+KEY"

    # 2) NLI
    if "nli" in stages and all(entails(ans, h) for h in hyps):
        return "pass", "nli", json.dumps(hyps, ensure_ascii=False)

    # 3) LLM
    if "llm" in stages:
        score, prompt = llm_judge(ans, hyps, q_text)
        return score, "llm", prompt

    # unresolved
    return "uncertain", "none", ""


# ─────────────────────────────────────────── interactive picker ──
def select_stages() -> set[str]:
    menu = """
Choose evaluation stages (comma-separated numbers or 'all'):

  [1] auto-fail regex
  [2] NLI entailment
  [3] LLM judge

Selection (default = all): """
    raw = input(menu).strip().lower()
    if raw in {"", "all"}:
        return {"auto", "nli", "llm"}
    mapping = {"1": "auto", "2": "nli", "3": "llm"}
    stages = {mapping[x] for x in raw.replace(",", " ").split() if x in mapping}
    if not stages:
        print("⚠️  Invalid choice – using all stages.")
        stages = {"auto", "nli", "llm"}
    return stages


# ──────────────────────────────────────────────────────── main ──
def main():
    cli = argparse.ArgumentParser(description="Score answers and tag method/query")
    cli.add_argument("--answers", required=True)
    cli.add_argument("--hypos",   required=True)
    cli.add_argument("--prompts", required=True)
    cli.add_argument("--out")
    args = cli.parse_args()

    stages  = select_stages()
    rows    = load_run_csv(args.answers)
    hypos   = load_hypos(args.hypos)
    prompts = load_prompts(args.prompts)

    for row in rows[1:]:  # skip header row if dataset re-included it
        score, method, qry = evaluate(row["prompt_id"],
                                      row["response"],
                                      prompts, hypos, stages)
        row["score"]        = score
        row["judge_method"] = method
        row["judge_query"]  = qry

    out = args.out or Path(args.answers).with_stem(
            Path(args.answers).stem + "_scored")
    pd.DataFrame(rows).to_csv(out, index=False, encoding="utf-8")
    print(f"✅ scores written to {out}")


if __name__ == "__main__":
    main()
