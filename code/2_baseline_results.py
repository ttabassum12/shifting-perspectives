import os
import sys
import datetime
import torch
import pandas as pd

from utils import bbq_axes, get_output
from transformers import AutoTokenizer, AutoModelForCausalLM
from dotenv import load_dotenv

load_dotenv()
hf_token = os.getenv("HF_TOKEN")
print(datetime.datetime.now())

if len(sys.argv) > 1:
    model_name = sys.argv[1]
else:
    raise ValueError("Model name must be provided as a command-line argument.")

# Map model names to short names
model_short_names = {
    "Qwen/Qwen2.5-7B-Instruct": "qwen",
    "meta-llama/Llama-3.1-8B-Instruct": "llama",
    "mistralai/Mistral-7B-Instruct-v0.1": "mistral",
}

model_short_name = model_short_names.get(model_name)
if not model_short_name:
    raise ValueError(f"Unknown model name: {model_name}")

# Load LLM
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token_id = tokenizer.eos_token_id
model = AutoModelForCausalLM.from_pretrained(model_name)

device = "cuda:0" if torch.cuda.is_available() else "cpu"
model = model.to(device)

## Get baseline first
def process_row(row):
    context = row["context"]
    question = row["question"]

    # Extract answers from the "answer_info" column
    answers = [row["ans0"], row["ans1"], row["ans2"]]
    
    input_text = (
        f"Answer with only a single letter.\n\n"
        f"The following are multiple choice questions (with answers):\n"
        f"{context} {question}\n"
        f"(a) {answers[0]}\n"
        f"(b) {answers[1]}\n"
        f"(c) {answers[2]}\n"
    )
    
    generated_answer = get_output(input_text, model, tokenizer).lower()
    
    predicted_label = -1
    for i, letter in enumerate(["a", "b", "c"]):
        if letter in generated_answer[:2]:
            predicted_label = i
            break
    if predicted_label == -1:
        answers = [row["ans0"], row["ans1"], row["ans2"]]
        for i, answer in enumerate(answers):
            if answer in generated_answer:
                predicted_label = i
                break

    correct = (predicted_label == row["label"])

    return pd.Series({
        "ans":         generated_answer,
        "prediction":  predicted_label,
        "correct":     correct
    })


all_dfs = []

for axis in bbq_axes:
    print(f"Evaluating {axis} at:", datetime.datetime.now())
    df = pd.read_csv(f"../data/bbq_test/{axis}_test.csv")

    df[['ans', 'prediction', 'correct']] = None
    df[['ans', 'prediction', 'correct']] = df.apply(process_row, axis=1)

    df['axis'] = axis
    all_dfs.append(df)

big_df = pd.concat(all_dfs, ignore_index=True)
output_dir = f'../results/{model_short_name}'
os.makedirs(output_dir, exist_ok=True)
big_df.to_csv(f'{output_dir}/bbq_baseline.csv', index=False)

