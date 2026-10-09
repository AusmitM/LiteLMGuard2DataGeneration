import pandas as pd

INPUT_SOURCE = "kalyannakka/Answerable-or-Not"
COL_ID = None                     
COL_PROMPT = "question"
COL_ANSWERABLE = "answerable"    
COL_CATEGORY = "specific_harms"   

def load_data(pilot=None):
    if INPUT_SOURCE.endswith(".csv"):
        df = pd.read_csv(INPUT_SOURCE)
    else:
        from datasets import load_dataset
        df = load_dataset(INPUT_SOURCE, split="train").to_pandas()
    if COL_ID is None or COL_ID not in df.columns:
        df["id"] = [f"p{i:05d}" for i in range(len(df))]
    else:
        df["id"] = df[COL_ID].astype(str)
    return df

def iterate_specific_harms(df, harm_arr):
    specific_harms = ""
    for _, row in df.iterrows():
        if specific_harms == "":
            specific_harms = row[COL_CATEGORY]
            harm_arr.append(specific_harms)
        elif specific_harms != row[COL_CATEGORY]:
            specific_harms = row[COL_CATEGORY]
            harm_arr.append(specific_harms)

df = load_data()
df_categorized = pd.read_csv("p2_run/merged.csv")
# print(df_categorized.head())
specific_harm_arr = []
iterate_specific_harms(df, specific_harm_arr)
random_prompts = []
random_prompts_answerable = []
random_prompts_non_answerable = []

for harm in specific_harm_arr:
    # taking the specific harm
    specificHarms = df[df['specific_harms'] == harm]
    # getting answerable
    answerable = specificHarms[specificHarms['answerable'] == "Yes"].sample(n = 2, random_state=42)
    a_id1 = answerable.iloc[0]['id']
    a_id2 = answerable.iloc[1]['id']
    if df_categorized[(df_categorized['id'] == a_id1)]['p2_final'].values[0] != "Benign" or df_categorized[(df_categorized['id'] == a_id2)]['p2_final'].values[0] != "Benign":
        print(f"Answerable id1: {a_id1} is not Benign")
        print(f"Answerable id2: {a_id2} is not Benign")
    random_prompts.append(answerable)
    random_prompts_answerable.append(answerable)
    # getting non-answerable
    non_answerable = specificHarms[specificHarms['answerable'] == "No"].sample(n =2, random_state=42)
    n_id1 = non_answerable.iloc[0]['id']
    n_id2 = non_answerable.iloc[1]['id']
    if df_categorized[(df_categorized['id'] == n_id1)]['p2_final'].values[0] != "Harmful Direct Request" or df_categorized[(df_categorized['id'] == n_id2)]['p2_final'].values[0] != "Harmful Direct Request":
        print(f"Non-Answerable id1: {n_id1} is not Harmful Direct Request")
        print(f"Non-Answerable id2: {n_id2} is not Harmful Direct Request")
    random_prompts.append(non_answerable)
    random_prompts_non_answerable.append(non_answerable)
    # print("Answerable:", answerable)
    # print("Non_Answerable:", non_answerable)

# we need to be sure that they are benign and Direct Harmful Reqest for each
combined_random_prompts = pd.concat(random_prompts)
combined_random_prompts.to_csv("random_prompts.csv", index=False)

combined_random_prompts_answerable = pd.concat(random_prompts_answerable)
combined_random_prompts_answerable.to_csv("random_prompts_answerable.csv", index=False)

combined_random_prompts_non_answerable = pd.concat(random_prompts_non_answerable)
combined_random_prompts_non_answerable.to_csv("random_prompts_non_answerable.csv", index=False) 
