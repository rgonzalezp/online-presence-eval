import asyncio
import json, yaml, re, argparse, pandas as pd, openai
from pathlib import Path
from openai import OpenAI
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch, torch.nn.functional as F
from extract_information_multi_turn_yml_version import select_models_to_run, load_model_configs
from models_multi  import create_models 
import braintrust
from openai import AsyncOpenAI

_JUDGE_MODELS = None            # type: dict[str, object]

def _get_judge_model(name: str = "judge_default"):
    """
    Return (and cache) a model from models_multi.create_models().
    The name must match an entry in configs/paid_models.yml.
    """
    global _JUDGE_MODELS
    if _JUDGE_MODELS is None:
        with open("configs/paid_models.yml", encoding="utf-8") as f:
            cfgs = yaml.safe_load(f)["models"]
        _JUDGE_MODELS = create_models(cfgs)
    if name not in _JUDGE_MODELS:
        raise KeyError(f"Judge-model '{name}' not found in paid_models.yml")
    return _JUDGE_MODELS[name]

# ───────────────────────────────────────────────────────── helpers ──
def load_api_keys():
    path = Path(__file__).parent / "configs" / "api_keys.yml"
    with open(path) as f:
        return yaml.safe_load(f)
API_KEYS = load_api_keys()
braintrust.login(api_key= API_KEYS["braintrust"]["api_key"])

def load_run_csv(path: str):
    df = pd.read_csv(path, encoding="utf-8")
    must = {"prompt_id", "response"}
    if not must.issubset(df.columns):
        raise ValueError("CSV must contain prompt_id and response")
    return df.to_dict(orient="records")


def load_hypos(path: str ):
    by_field = {}
    with open(path, encoding="utf-8") as f:
        for e in json.load(f):
            by_field.setdefault(e["field"], []).append(e["hypothesis"])
    return by_field


def load_prompts(path: str = "data/grounding/ultra_specific_prompts_mapping.yml"):
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

## Auto fail regex is crap it only works for one case (affiliation, and it is not well implemented)
NEG     = re.compile(r"\b(no|not|never|formerly|previously|ex-|until)\b", re.I)
KEY_ENT = re.compile(r"\bcornell (university|tech)\b", re.I)

def auto_fail_regex(ans: str) -> bool:
    a = ans.lower()
    return bool(NEG.search(a) and KEY_ENT.search(a))


# ────────────────────────────────────────────────────────── NLI  ──
nli_model = "MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli"
_tok = AutoTokenizer.from_pretrained(nli_model)
_nli = AutoModelForSequenceClassification.from_pretrained(
        nli_model).eval()

def entails(premise: str, hyp: str, thr: float = .800) -> bool:
    pair = _tok(premise, hyp, return_tensors="pt", truncation=True)
    with torch.no_grad():
        probs = F.softmax(_nli(**pair).logits, dim=-1)

    return probs[0,0].item() >= thr


# ─────────────────────────────────────────────── GPT-judge prompt ──

PROMPT = """\
You are comparing a submitted answer to an expert answer on a given question. Here is the data:
[BEGIN DATA]
************
[Question]: {input}
************
[Expert]: {expected}
************
[Submission]: {output}
************
[END DATA]

Compare the factual content of the submitted answer with the expert answer. Ignore any differences in style, grammar, or punctuation.
The submitted answer may either be a subset or superset of the expert answer, or it may conflict with it. Determine which case applies. Answer the question by selecting one of the following options:
(A) The submitted answer is a subset of the expert answer and is fully consistent with it.
(B) The submitted answer is a superset of the expert answer and is fully consistent with it.
(C) The submitted answer contains all the same details as the expert answer.
(D) There is a disagreement between the submitted answer and the expert answer.
(E) The answers differ, but these differences don't matter from the perspective of factuality.

Answer the question by calling `select_choice` with your reasoning in a step-by-step manner to be
sure that your conclusion is correct. Avoid simply stating the correct answer at the outset. Select a
single choice by setting the `choice` parameter to a single choice from A, B, C, D, or E.
"""

# Since we're testing for hallucinations, penalize (B) as much as (D).
CHOICE_SCORES = {
    "A": 0.5,
    "B": 0,
    "C": 1,
    "D": 0,
    "E": 1,
}

async def classifier(input, output, expected):
    
    client = braintrust.wrap_openai(AsyncOpenAI(api_key=API_KEYS["openai"]["api_key"]))

    response = await client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "user",
                "content": PROMPT.format(input=input, output=output, expected=expected),
            }
        ],
        temperature=0,
        tools=[
            {
                "type": "function",
                "function": {
                    "name": "rate",
                    "description": "Call this function to select a choice.",
                    "parameters": {
                        "properties": {
                            "reasons": {
                                "description": "Write out in a step by step manner your reasoning to be sure that your conclusion is correct. Avoid simply stating the correct answer at the outset.",
                                "type": "string",
                            },
                            "choice": {
                                "description": "The choice",
                                "type": "string",
                                "enum": ["A", "B", "C", "D", "E"],
                            },
                        },
                        "required": ["reasons", "choice"],
                        "type": "object",
                    },
                },
            }
        ],
        tool_choice={"type": "function", "function": {"name": "rate"}},
    )
    arguments = json.loads(response.choices[0].message.tool_calls[0].function.arguments)
    print(arguments)
    choice = arguments.get("choice")          # returns None if key isn’t present
    if choice is None:
        choice = arguments["reasons"]
    ## check for attribute
    return choice, CHOICE_SCORES[choice] if choice in CHOICE_SCORES else None, PROMPT.format(input=input, output=output, expected=expected)

def llm_judge(ans: str, hyps: list[str], q: str,
              selected_models: dict ) -> tuple[str, str]:
    """
    Use the unified wrapper in models_multi to grade with an arbitrary provider.
    Returns (one-word verdict, prompt_sent_to_llm).
    """
    prompt = f"""
You are comparing a submitted answer to a set of ground-truths claims. Here is the data:

[BEGIN DATA]
************
[Question]:\"{q}\"
************
[Ground-truth claims]:
{json.dumps(hyps, ensure_ascii=False, indent=2)}
************
[Submission]:\"{ans}\"
************
[END DATA]

Compare the factual content of the submitted answer with the expert answer. Ignore any differences in style, grammar, or punctuation.
The submitted answer may either be a subset or superset of the expert answer, or it may conflict with it. Determine which case applies. Answer the question by selecting one of the following options:
(A) The submitted answer is a subset of the expert answer and is fully consistent with it.
(B) The submitted answer is a superset of the expert answer and is fully consistent with it.
(C) The submitted answer contains all the same details as the expert answer.
(D) There is a disagreement between the submitted answer and the expert answer.
(E) The answers differ, but these differences don't matter from the perspective of factuality.

Answer the question by calling `select_choice` with your reasoning in a step-by-step matter to be
sure that your conclusion is correct. Avoid simply stating the correct answer at the outset. Select a
single choice by setting the `choice` parameter to a single choice from A, B, C, D, or E.
""".strip()
    

    judge_model = _get_judge_model(selected_models[0])
    text, _meta, _raw, _hist = judge_model.generate(prompt, mode="no-search")
    verdict = text.strip().split()[0].lower()        # keep first token
    return verdict, prompt, selected_models[0]


# ────────────────────────────────────────────── evaluate answer ──
async def evaluate(prompt_id: str, ans: str, prompts: dict,
             hypos_by_field: dict, stages: set[str], selected_models: dict) -> tuple[str, str, str]:
    """
    Returns (score, method, query)
    """
    fields = prompts[prompt_id]["fields"]
    hyps   = [h for f in fields for h in hypos_by_field.get(f, [])]
    q_text = prompts[prompt_id]["question"]

    # 1) auto-fail
    if "auto" in stages and auto_fail_regex(ans):
        return "fail", "auto", "NEG+KEY", "regex"

    # 2) NLI
    if "nli" in stages and all(entails(ans, h) for h in hyps):
        return "entails", "nli", json.dumps(hyps, ensure_ascii=False), "DeBERTa"

    # 3) LLM
    if "llm" in stages:
        grade,score,prompt = await classifier(q_text, ans, hyps )
        return grade + " " + str(score), "llm", prompt, "open_ai"

    # unresolved
    return "uncertain", "none", "",""


# ─────────────────────────────────────────── interactive picker ──
def select_stages() -> set[str]:
    menu = """
Choose evaluation stages (comma-separated numbers or 'all'):

  [1] auto-fail regex (disabled)
  [2] NLI entailment
  [3] LLM judge

Selection (default = all): """
    raw = input(menu).strip().lower()
    if raw in {"", "all"}:
        #return {"auto", "nli", "llm"}
        return {"nli", "llm"}
    mapping = { "2": "nli", "3": "llm"}
    stages = {mapping[x] for x in raw.replace(",", " ").split() if x in mapping}
    if not stages:
        print("⚠️  Invalid choice – using all stages.")
        stages = {"nli", "llm"}
    return stages


# ──────────────────────────────────────────────────────── main ──
async def main():
    cli = argparse.ArgumentParser(description="Score answers and tag method/query")
    cli.add_argument("--answers", required=True)
    cli.add_argument("--hypos",   required=False)
    cli.add_argument("--prompts", required=False)
    cli.add_argument("--out")
    args = cli.parse_args()


    

    stages  = select_stages()
    rows    = load_run_csv(args.answers)
    hypos   = load_hypos(args.hypos or "data/grounding/ricardo_hypotheses.json")
    prompts = load_prompts(args.prompts or "data/grounding/ultra_specific_prompts_mapping.yml")

    cfgs = load_model_configs()
    models = create_models(cfgs)
    selected_models = select_models_to_run(models, None)

    for row in rows[1:]:  # skip header row if dataset re-included it
        score, method, qry, model_name = await evaluate(row["prompt_id"],
                                      row["response"],
                                      prompts, hypos, stages, selected_models)
        row["score"]        = score
        row["judge_method"] = method
        row["judge_query"]  = qry
        row["judge_name"]  = model_name

    out = args.out or Path(args.answers).with_stem(
            Path(args.answers).stem + "_scored")
    pd.DataFrame(rows).to_csv(out, index=False, encoding="utf-8")
    print(f"✅ scores written to {out}")

if __name__ == "__main__":
    asyncio.run(main())
