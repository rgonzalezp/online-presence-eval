from models import create_models
from toy_examples.extract_information_multi_turn import (
    load_prompts,
    load_model_configs,
    run_models,
    save_results
)

def main():
    # 1) Load prompts
    prompts_df = load_prompts()
    
    # 2) Load model configs
    model_configs = load_model_configs()
    
    # 3) Create models
    models = create_models(model_configs)
    
    # 4) Run test prompt through models
    test_row = prompts_df.iloc[1]
    prompt, mode = test_row["prompt"], test_row["mode"]
    
    results = run_models(models, prompt, mode)
    
    # 5) Save results
    save_results(results)

if __name__ == "__main__":
    main()