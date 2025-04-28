import pandas as pd
import yaml
import sys
from models import create_models
from utils.api_logger import log_api_call
from utils.refusal_detector import detect_refusal, _MODEL as _REFUSAL_MODEL

def load_prompts(file_path="prompt_permutations_fixed_eval.csv"):
    """Load prompts from a CSV file."""
    return pd.read_csv(file_path)

def load_model_configs(file_path="configs/paid_models.yml"):
    """Load model configurations from a YAML file."""
    with open(file_path) as f:
        return yaml.safe_load(f)["models"]

def display_available_models(models):
    """Display available models with index numbers."""
    print("\nAvailable models:")
    for i, name in enumerate(models.keys(), 1):
        print(f"  [{i}] {name}")
    print()

def display_available_prompts(prompts_df):
    """Display available prompts with index numbers and their modes."""
    print("\nAvailable prompts:")
    for i, row in prompts_df.iterrows():
        print(f"  [{i+1}] Mode: {row['mode']}")
        print(f"      Job: {row['job_variant']}")
        print(f"      Name: {row['name_variant']}")
        print()
    print()

def select_models_to_run(models, selection=None):
    """Select which models to run based on user input."""
    model_names = list(models.keys())
    
    if not selection:
        # Interactive mode
        display_available_models(models)
        selection = input("Enter models to run (numbers without separators, provider name, or 'all'): ")
    
    if selection.lower() == 'all':
        return model_names
    
    # Check if selection is provider-based
    providers = {'openai', 'anthropic', 'gemini', 'grok'}
    if selection.lower() in providers:
        provider = selection.lower()
        return [name for name in model_names if name.lower().startswith(provider)]
    
    # Check if selection is number-based
    try:
        indices = [int(digit) for digit in selection if digit.isdigit()]
        return [model_names[idx-1] for idx in indices if 1 <= idx <= len(model_names)]
    except (ValueError, IndexError):
        print(f"Invalid selection: {selection}")
        return []

def select_prompts_to_run(prompts_df, selection=None):
    """Select which prompts to run based on user input."""
    if not selection:
        # Interactive mode
        display_available_prompts(prompts_df)
        selection = input("Enter prompts to run (numbers without separators, 'all', 'web-search', or 'no-search'): ")
    
    if selection.lower() == 'all':
        return prompts_df
    
    # Check if selection is mode-based
    if selection.lower() in ['web-search', 'no-search']:
        return prompts_df[prompts_df['mode'] == selection.lower()]
    
    # Check if selection is number-based
    try:
        indices = [int(digit) for digit in selection if digit.isdigit()]
        valid_indices = [idx-1 for idx in indices if 0 <= idx-1 < len(prompts_df)]
        return prompts_df.iloc[valid_indices]
    except (ValueError, IndexError):
        print(f"Invalid selection: {selection}")
        return pd.DataFrame()

def run_models(models, prompts_df, selection=None):
    """Run selected prompts through selected models and collect results."""
    results = []
    
    # Determine which models to run
    selected_models = select_models_to_run(models, selection)
    
    if not selected_models:
        print("No models selected to run.")
        return results
    
    # Determine which prompts to run
    selected_prompts = select_prompts_to_run(prompts_df)
    
    if selected_prompts.empty:
        print("No prompts selected to run.")
        return results
    
    print(f"\nRunning {len(selected_prompts)} prompts through {len(selected_models)} models...")
    
    for _, prompt_row in selected_prompts.iterrows():
        prompt, mode = prompt_row["prompt"], prompt_row["mode"]
        print(f"\n=== Running prompt with mode: {mode} ===")
        
        for model_name in selected_models:
            model = models[model_name]
            print(f"\nRunning {model_name}...")
            try:
                text, metadata, responseapi = model.generate(prompt, mode)
            except Exception as e:
                text = f"[ERROR] {e}"
                metadata = {}
                responseapi = {}

            refusal_label, refusal_conf = detect_refusal(user_query=prompt, text_response=text)

            print(
                f"Refusal detection: {refusal_label} "
                f"(Class {_REFUSAL_MODEL.config.label2id[refusal_label]}), "
                f"Confidence: {refusal_conf:.4f}"
            )

            log_api_call(
                model_name,
                mode,
                prompt_row["job_variant"],
                prompt_row["name_variant"],
                prompt,
                text,
                metadata,
                responseapi
            )

            # Register into results
            results.append({
                "model": model_name,
                "mode": mode,
                "job_variant": prompt_row["job_variant"],
                "name_variant": prompt_row["name_variant"],
                "prompt": prompt,
                "response": text,
                "metadata": metadata,
                "responseapi": responseapi
            })

            # Print output
            print(f"\n--- {model_name} ({mode}) ---")
            print(f"{text[:200]}..." if len(text) > 200 else text)
    
    return results

def save_results(results, file_path="registered_responses.csv"):
    """Save results to a CSV file."""
    results_df = pd.DataFrame(results)
    results_df.to_csv(file_path, index=False)
    print(f"✅ All responses saved to {file_path}")

def main(model_selection=None, prompt_selection=None):
    """Run the main prompt evaluation workflow."""
    # 1) Load prompts
    prompts_df = load_prompts()
    
    # 2) Load model configs
    model_configs = load_model_configs()
    
    # 3) Create models
    models = create_models(model_configs)
    
    # 4) Run prompts through models
    results = run_models(models, prompts_df, model_selection)
    
    # 5) Save results
    if results:
        save_results(results)

if __name__ == "__main__":
    # Get selections from command line args if provided
    model_selection = ''.join(sys.argv[1:2]) if len(sys.argv) > 1 else None
    prompt_selection = ''.join(sys.argv[2:3]) if len(sys.argv) > 2 else None
    main(model_selection, prompt_selection) 