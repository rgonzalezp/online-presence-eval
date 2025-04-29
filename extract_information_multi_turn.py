import pandas as pd
import yaml
import sys
from datetime import datetime
from models_multi import create_models
from utils.api_logger import log_api_call
from utils.refusal_detector import detect_refusal, _MODEL as _REFUSAL_MODEL


def load_followup_prompts(file_path="data/overview_prompts_adjusted.csv"):
    """Load follow-up prompts from a CSV file."""
    return pd.read_csv(file_path)


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
    if 'mode' in prompts_df.columns and \
       {'prompt', 'mode', 'job_variant', 'name_variant'}.issubset(prompts_df.columns):
        # detailed CSV
        selected_prompts = select_prompts_to_run(prompts_df, selection)
    else:
        # simple one-column CSV
        selected_prompts = select_simple_prompts_to_run(prompts_df, selection)

    if selected_prompts.empty:
        print("No prompts selected.")
        return []

    # initialize per-model history
    if histories is None:
        histories = {name: None for name in selected_models}
    results = []

    print(f"\nRunning {len(selected_prompts)} prompts through {len(selected_models)} models...")
    for _, row in selected_prompts.iterrows():
        prompt      = row['prompt']
        # detect whether this is a "starter" (detailed) or "follow-up" (simple) prompt
        is_detailed = 'mode' in row and 'job_variant' in row and 'name_variant' in row

        prompt       = row['prompt']
        mode         = row.get('mode', 'no-search')
        job_variant  = row.get('job_variant', '')
        name_variant = row.get('name_variant', '')
        
        if is_detailed:
            mode        = row['mode']
            job_variant = row['job_variant']
            name_variant= row['name_variant']
            print(f"\n=== Starter Prompt ({mode}): {job_variant} | {name_variant} ===")
        else:
            # follow-ups have no mode/variants
            print(f"\n=== Follow-up Prompt: {prompt} ===")

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
                'job_variant':      job_variant,
                'name_variant':     name_variant,
                'prompt':           prompt,
                'response':         text,
                'metadata':         metadata,
                'responseapi':      raw,
                'refusal_label':    refusal_label,
                'refusal_confidence': refusal_conf
            })

            print(f"Response: {text[:200]}{'...' if len(text) > 200 else ''}")

    return results, histories


def save_results(results, file_path="registered_responses.csv"):
    df = pd.DataFrame(results)
    df.to_csv(file_path, index=False)
    print(f"✅ All responses saved to {file_path}")


def main(model_selection=None, prompt_selection=None):
    # load and run starter
    starter_df = load_starter_prompts()
    cfgs = load_model_configs()
    models = create_models(cfgs)
    starter_results, histories = run_models(models, starter_df, model_selection)
    # filter non-refusers
    non_refusers = {r['model'] for r in starter_results if r['refusal_label']=='Non-refusal'}
    # load and run follow-ups
    followups_df = load_followup_prompts().iloc[0:6]  # Select only the first row
    cont_models = {m:models[m] for m in non_refusers}
    experiment_results = []
    if cont_models:
        experiment_results, histories = run_models(cont_models, followups_df,
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
