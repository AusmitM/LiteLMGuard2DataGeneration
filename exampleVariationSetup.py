import pandas as pd

# loading the csv
answerable = pd.read_csv("randomlyPickedPrompts/random_prompts_answerable.csv")
non_answerable = pd.read_csv("randomlyPickedPrompts/random_prompts_non_answerable.csv")
merged = pd.read_csv("p2_run/merged.csv")

# Added prompt strategy
answerable_prompt_strategy = ["Benign", "Contextualized/Educational Framing", "Roleplay", "Hypothetical"]
non_answerable_prompt_strategy = ["Harmful Direct Request", "Contextualized/Educational Framing", "Roleplay", "Hypothetical", "Indirect/Multistep", "Obfuscated", "Instruction Override", "Adversarial Attack", "Prompt Injection"]

# updating answerable csv
answerable_variation_example = answerable[['risk_area', 'types_of_harm', 'specific_harms', 'answerable', 'question']].copy()
answerable_variation_example['indexReference'] = answerable['id'] 
answerable_variation_example['promptStrategy'] = "Benign"
all_answerable_prompt_strategy  = []
for index, row in answerable_variation_example.iterrows():
    all_answerable_prompt_strategy.append(row.to_dict())

    for prompt in answerable_prompt_strategy:
        if prompt != "Benign":
            all_answerable_prompt_strategy.append({'risk_area': row['risk_area'], 'types_of_harm': row['types_of_harm'], 'specific_harms': row['specific_harms'], 'answerable': None, 'question': None, 'indexReference': row['indexReference'], 'promptStrategy': prompt})
answerable_df = pd.DataFrame(all_answerable_prompt_strategy)
answerable_df.to_csv("randomlyPickedPrompts/variationExamples/answerable_variation_example.csv", index=False)


# updating non-answerable csv
non_answerable_variation_example = non_answerable[['risk_area', 'types_of_harm', 'specific_harms', 'answerable', 'question']].copy()
non_answerable_variation_example['indexReference'] = non_answerable['id']
non_answerable_variation_example['promptStrategy'] = "Harmful Direct Request"
all_non_answerable_prompt_strategy  = []
for index, row in non_answerable_variation_example.iterrows():
    all_non_answerable_prompt_strategy.append(row.to_dict())

    for prompt in non_answerable_prompt_strategy:
        if prompt != "Harmful Direct Request":
            all_non_answerable_prompt_strategy.append({'risk_area': row['risk_area'], 'types_of_harm': row['types_of_harm'], 'specific_harms': row['specific_harms'], 'answerable': None, 'question': None, 'indexReference': row['indexReference'], 'promptStrategy': prompt})
non_answerable_df = pd.DataFrame(all_non_answerable_prompt_strategy)
non_answerable_df.to_csv("randomlyPickedPrompts/variationExamples/non_answerable_variation_example.csv", index=False)
