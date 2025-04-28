import os
import pandas as pd
from jinja2 import Environment, FileSystemLoader
import itertools
import random


script_dir = os.path.dirname(os.path.abspath(__file__))
template_dir = os.path.join(script_dir)

SHEET_ID = "1F1kWBR1eZ_X6DLz1b1cIZLhROOiDGYZjPUFdoGVvezo"

names_url = f"https://docs.google.com/spreadsheets/d/{SHEET_ID}/export?format=csv&gid=1859748400"
jobs_url  = f"https://docs.google.com/spreadsheets/d/{SHEET_ID}/export?format=csv&gid=1512071086"

names_df = pd.read_csv(names_url)
jobs_df  = pd.read_csv(jobs_url)

modes    = ["no-search", "web-search"]


env = Environment(
    loader=FileSystemLoader(template_dir),
    autoescape=True
)
template = env.get_template("person_identify_starter_template.j2")
# 4. Render all prompt permutations
records = []
for mode in ["no-search", "web-search"]:
    all_combos = list(itertools.product([mode], jobs_df["job_variant"], names_df["name_variant"]))
    sampled = random.sample(all_combos, 3)
    for mode_, job, name in sampled:
        prompt_text = template.render(
            mode=mode_,
            job_variant=job,
            name_variant=name
        )
        records.append({
            "mode":        mode_,
            "job_variant": job,
            "name_variant":name,
            "prompt":      prompt_text
        })

prompts_df = pd.DataFrame(records)

# 5. Save to CSV in prompts directory
output_path = os.path.join(script_dir, "identify_person_starter_prompts_randomized_eval.csv")
prompts_df.to_csv(output_path, index=False)

# 6. Preview the first few prompts
prompts_df.head()
