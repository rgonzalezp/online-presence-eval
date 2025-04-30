import pandas as pd
import yaml
import sys
from datetime import datetime
from models_multi import create_models
from utils.api_logger import log_api_call
from utils.refusal_detector import detect_refusal, _MODEL as _REFUSAL_MODEL
import os


def load_followup_prompts(file_path="data/grounding/ultra_specific_prompts_mapping.yml"):
    """Load follow-up prompts (with id, question, fields, phase) from YAML."""
    with open(file_path, encoding="utf-8") as f:
        prompts = yaml.safe_load(f)

    return prompts

def select_followups_to_run(followups, selection=None):
    """
    Interactive selector for follow-up prompts (list of dicts).
    * 'all'  – return entire list
    * comma-/space-separated numbers – return those indices (1-based)
    * otherwise treat input as an id-substring filter.
    """
    if not selection:
        print("\nAvailable follow-up prompts:")
        for i, p in enumerate(followups, 1):
            print(f"  [{i}] {p['id']:<25} | {p['question'][:60]}…")
        selection = input("Enter prompts to run (numbers, id substring, or 'all'): ")

    sel_lower = selection.lower()

    # full list
    if sel_lower == "all":
        return followups

    # numeric picks (e.g. "1 3 5")
    if all(tok.isdigit() for tok in sel_lower.split()):
        idx = [int(tok) - 1 for tok in sel_lower.split()]
        return [followups[i] for i in idx if 0 <= i < len(followups)]

    # substring match on id
    filtered = [p for p in followups if sel_lower in p["id"].lower()]
    if filtered:
        return filtered

    print(f"Invalid follow-up selection: {selection}")
    return []

def load_starter_prompts(file_path="prompts/identify_person_starter_prompts_fixed_eval.csv"):
    """Load starter prompts from a CSV file."""
    return pd.read_csv(file_path)


def load_model_configs(file_path="configs/paid_models.yml"):
    """Load model configurations from a YAML file."""
    with open(file_path) as f:
        return yaml.safe_load(f)["models"]


def select_models_to_run(models, selection=None):
    """Select which models to run based on user input."""
    model_names = list(models.keys())
    if not selection:
        print("\nAvailable models:")
        for i, name in enumerate(model_names, 1):
            print(f"  [{i}] {name}")
        selection = input("Enter models to run (numbers, provider, or 'all'): ")
    if selection.lower() == 'all':
        return model_names
    providers = {'openai', 'anthropic', 'gemini', 'grok'}
    if selection.lower() in providers:
        return [n for n in model_names if n.lower().startswith(selection.lower())]
    try:
        indices = [int(d) for d in selection if d.isdigit()]
        return [model_names[i-1] for i in indices]
    except:
        print(f"Invalid model selection: {selection}")
        return []

def select_prompts_to_run(prompts_df, selection=None):
    """Select which detailed prompts to run (expects columns: prompt, mode, job_variant, name_variant)."""
    if not selection:
        print("\nAvailable prompts:")
        for i, row in prompts_df.iterrows():
            print(f"  [{i+1}] {row['mode']} | {row['job_variant']} | {row['name_variant']}")
        selection = input("Enter prompts to run (numbers, 'all', 'web-search', or 'no-search'): ")

    if selection.lower() == 'all':
        return prompts_df

    if selection.lower() in ['web-search', 'no-search']:
        return prompts_df[prompts_df['mode'] == selection.lower()]

    try:
        # allow things like "1 3 5"
        picks = [int(x) - 1 for x in selection.split() if x.isdigit()]
        return prompts_df.iloc[picks]
    except:
        print(f"Invalid prompt selection: {selection}")
        return pd.DataFrame()


def select_simple_prompts_to_run(prompts_df, selection=None):
    """
    Select which simple prompts to run (DataFrame has exactly one column of questions).
    Signature matches select_prompts_to_run.
    """
    df = prompts_df.copy()
    # rename that single column to 'prompt'

    if not selection:
        print("\nAvailable prompts:")
        for i, prompt in enumerate(df['prompt'], 1):
            print(f"  [{i}] {prompt}")
        selection = input("Enter prompts to run (numbers separated by spaces, or 'all'): ")

    if selection.lower() == 'all':
        return df

    try:
        picks = [int(x) - 1 for x in selection.split() if x.isdigit()]
        return df.iloc[picks]
    except:
        print(f"Invalid prompt selection: {selection}")
        return pd.DataFrame(columns=['prompt'])



def run_models(models, prompts_df, selection=None, histories=None):
    """Run prompts through models, maintaining per-model histories and collecting results."""
    # choose models
    selected_models = select_models_to_run(models, selection)
    if not selected_models:
        print("No models selected.")
        return [], histories or {}

    # decide which selector to use
    

    if isinstance(prompts_df, list):
        selected_prompts = prompts_df
    else:
        if 'mode' in prompts_df.columns and \
            {'prompt', 'mode', 'job_variant', 'name_variant'}.issubset(prompts_df.columns):
        # detailed CSV
            selected_prompts = select_prompts_to_run(prompts_df, selection)
        else:
        # simple one-column CSV
            selected_prompts = select_simple_prompts_to_run(prompts_df, selection)

    is_empty = (
        (isinstance(selected_prompts, list) and len(selected_prompts) == 0) or
        (hasattr(selected_prompts, "empty") and selected_prompts.empty)
    )
    if is_empty:
        print("No prompts selected.")
        return []

    # initialize per-model history
    if histories is None:
        histories = {name: None for name in selected_models}
    results = []

    print(f"\nRunning {len(selected_prompts)} prompts through {len(selected_models)} models...")
    rows_iter = selected_prompts if isinstance(selected_prompts, list) else selected_prompts.itertuples(index=False)

    for row in rows_iter:
        if isinstance(row, dict):                            # follow-up
            prompt_id    = row['id']
            prompt       = row['question']
            is_detailed  = False
            mode         = 'no-search'
            job_variant  = name_variant = ''
            print(f"\n=== Follow-up Prompt: {prompt} (id={prompt_id}) ===")
        else:                                                # starter (DataFrame row)
            prompt_id    = row.name_variant or row.prompt    # keep some id
            prompt       = row.prompt
            is_detailed  = hasattr(row, 'mode')
            mode         = getattr(row, 'mode', 'no-search')
            job_variant  = getattr(row, 'job_variant', '')
            name_variant = getattr(row, 'name_variant', '')
            print(f"\n=== Starter Prompt ({mode}): {job_variant} | {name_variant} ===")

        

        for name in selected_models:
            model = models[name]

            if histories[name] is None:
             # first turn: just the prompt string
                messages = [{"role":"user","content":prompt}]
            else:
                messages = histories[name] + [{"role":"user","content":prompt}]
            print(f"\n-> {name}")
            try:
                text, metadata, raw, new_hist = model.generate(messages, mode)
            except Exception as e:
                text, metadata, raw, new_hist = f"[ERROR] {e}", {}, {}, None

            histories[name] = new_hist

            # refusal detection
            refusal_label, refusal_conf = detect_refusal(user_query=prompt, text_response=text)
            print(
                f"Refusal detection: {refusal_label} "
                f"(Class {_REFUSAL_MODEL.config.label2id[refusal_label]}), "
                f"Confidence: {refusal_conf:.4f}"
            )

            # log call
            log_api_call(name, mode, job_variant, name_variant, prompt, text, metadata, raw,  history=new_hist)

            # record
            results.append({
                'model':            name,
                'mode':             mode,
                'prompt_id':        prompt_id,
                'job_variant':      job_variant,
                'name_variant':     name_variant,
                'prompt':           prompt,
                'response':         text,
                'refusal_label':    refusal_label,
                'refusal_confidence': refusal_conf,
                'metadata':         metadata,
                'responseapi':      raw,
            })

            print(f"Response: {text[:200]}{'...' if len(text) > 200 else ''}")

    return results, histories


def save_results(results, file_path="registered_responses.csv"):
    # ensure the experiments directory exists
    os.makedirs("data/experiments", exist_ok=True)

    # prefix the file path so it goes under data/experiments
    full_path = os.path.join("data", "experiments", file_path)

    df = pd.DataFrame(results)
    df.to_csv(full_path , index=False)
    print(f"✅ All responses saved to {full_path }")


def main(model_selection=None, prompt_selection=None):
    # load and run starter
    starter_df = load_starter_prompts()
    cfgs = load_model_configs()
    models = create_models(cfgs)
    starter_results, histories = run_models(models, starter_df, model_selection)
    # filter non-refusers
    non_refusers = {r['model'] for r in starter_results if r['refusal_label']=='Non-refusal'}
    # load and run follow-ups

    followups_list = load_followup_prompts() 
    selected_followups = select_followups_to_run(followups_list, prompt_selection)
    print(selected_followups)
    cont_models = {m:models[m] for m in non_refusers}
    experiment_results = []
    if cont_models:
        experiment_results, histories = run_models(cont_models, selected_followups,
                                         histories=histories)
    # save combined
    all_results = starter_results + experiment_results
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M")
    save_results(all_results, file_path= f"full_experiment_results{timestamp}.csv")
    print(f"\n✅ Driver complete — results in full_experiment_results.csv (Generated at {timestamp})")

if __name__ == '__main__':
    sel1 = ''.join(sys.argv[1:2]) if len(sys.argv)>1 else None
    sel2 = ''.join(sys.argv[2:3]) if len(sys.argv)>2 else None
    main(sel1, sel2)
