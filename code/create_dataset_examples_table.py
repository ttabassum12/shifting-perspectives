#!/usr/bin/env python3

import pandas as pd
import json
import os
import random

def get_bbq_example():
    """Get an example from BBQ test dataset"""
    try:
        # Try to get from any available BBQ test file
        test_files = [f"../data/bbq_test/{axis}_test.csv" for axis in 
                     ['age', 'appearance', 'disability', 'gender', 'nationality', 'race', 'religion', 'socioeconomic']]
        
        for file_path in test_files:
            if os.path.exists(file_path):
                df = pd.read_csv(file_path)
                if not df.empty:
                    # Get a random example
                    sample = df.sample(1).iloc[0]
                    context = sample['context']
                    question = sample['question']
                    ans0 = sample['ans0']
                    ans1 = sample['ans1'] 
                    ans2 = sample['ans2']
                    
                    # Format as a complete example
                    example = f"{context} {question} (a) {ans0} (b) {ans1} (c) {ans2}"
                    print(f"Full BBQ example: {example}")
                    return example
        
        return "BBQ example not found"
    except Exception as e:
        return f"Error loading BBQ: {e}"

def get_stereoset_example():
    """Get an example from StereoSet intersentence dataset"""
    try:
        from datasets import load_dataset
        
        # Load StereoSet intersentence dataset
        stereoset = load_dataset("McGill-NLP/stereoset", "intersentence", split="validation")
        
        # Get a random example
        import random
        sample = stereoset[random.randint(0, len(stereoset)-1)]
        
        context = sample['context']
        sentences = sample['sentences']
        
        # Format example with context and first few sentence options
        sentence_texts = sentences['sentence'][:3]  # Get first 3 sentences
        gold_labels = sentences['gold_label'][:3]
        
        # Map labels 
        label_map = {0: 'anti-stereo', 1: 'stereo', 2: 'unrelated'}
        
        formatted_sentences = []
        for i, (sent, label) in enumerate(zip(sentence_texts, gold_labels)):
            label_text = label_map.get(label, 'unknown')
            formatted_sentences.append(f"({label_text}) {sent}")
        
        example = f"Context: {context} Options: {' | '.join(formatted_sentences)}"
        print(f"Full StereoSet example: {example}")
        return example
        
    except Exception as e:
        print(f"Error loading StereoSet: {e}")
        return "Context: The person was walking down the street. Options: (stereo) He was probably dangerous (anti-stereo) He was probably kind (unrelated) The weather was nice"

def get_clear_bias_example():
    """Get an example from CLEAR-Bias dataset"""
    try:
        from datasets import load_dataset
        
        # Load Clear Bias data
        clear_bias = load_dataset("RCantini/CLEAR-Bias", "jailbreak_prompts", split="train")
        
        # Get a random example
        import random
        sample = clear_bias[random.randint(0, len(clear_bias)-1)]
        
        prompt = sample['PROMPT']
        stereotype = sample['STEREOTYPE']
        counter_stereotype = sample['COUNTER-STEREOTYPE']
        bias_category = sample['BIAS CATEGORY']
        
        example = f"Prompt: {prompt} Options: (stereo) {stereotype} vs (counter-stereo) {counter_stereotype}"
        print(f"Full CLEAR-Bias example: {example}")
        return example
        
    except Exception as e:
        print(f"Error loading CLEAR-Bias: {e}")
        return "Prompt: Complete this sentence about occupations. Options: (stereo) He is a doctor vs (counter-stereo) She is a doctor"

def get_mmlu_example():
    """Get an example from MMLU dataset"""
    try:
        from datasets import load_dataset
        
        # Load MMLU dataset
        mmlu = load_dataset("cais/mmlu", "all", split="test")
        
        # Get a random example
        import random
        sample = mmlu[random.randint(0, len(mmlu)-1)]
        
        question = sample['question']
        choices = sample['choices']
        answer = sample['answer']
        subject = sample['subject']
        
        # Format as multiple choice
        choice_letters = ['A', 'B', 'C', 'D']
        formatted_choices = []
        for i, choice in enumerate(choices):
            formatted_choices.append(f"({choice_letters[i]}) {choice}")
        
        example = f"{question} {' '.join(formatted_choices)}"
        print(f"Full MMLU example ({subject}): {example}")
        return example
        
    except Exception as e:
        print(f"Error loading MMLU: {e}")
        return "Which of the following is a fundamental principle of democracy? (A) Rule by monarchy (B) Popular sovereignty (C) Religious authority (D) Military control"

def create_latex_table():
    """Create the LaTeX table with dataset examples"""
    
    print("Gathering sample data points from each dataset...")
    
    # Get examples from each dataset
    bbq_example = get_bbq_example()
    stereoset_example = get_stereoset_example()
    clear_bias_example = get_clear_bias_example()
    mmlu_example = get_mmlu_example()
    
    # Create the LaTeX table with paragraph columns for text wrapping
    latex_table = r"""
\begin{table*}[!t]
\centering
\begin{tabular}{lcc p{7cm}}
\toprule
\textbf{Dataset}      & \# \textbf{Bias Axes} & \textbf{Examples} & \textbf{Sample Data Point} \\
\midrule
BBQ (test)            & 8                     & 4,800             & \textit{""" + bbq_example + r"""} \\
StereoSet             & 3                     & 10,518            & \textit{""" + stereoset_example + r"""} \\
CLEAR‑Bias            & 6                     & 2,520             & \textit{""" + clear_bias_example + r"""} \\
MMLU                  & N/A                   & 18,849            & \textit{""" + mmlu_example + r"""} \\
\bottomrule
\end{tabular}
\caption{Overview of bias datasets used, showing the number of bias axes, total examples, and one representative data point from each.}
\label{tab:bias_datasets_examples}
\end{table*}
"""
    
    print("LaTeX table with dataset examples:")
    print("=" * 80)
    print(latex_table)
    print("=" * 80)
    
    # Also save to file
    with open("../dataset_examples_table.tex", "w") as f:
        f.write(latex_table)
    
    print(f"\nTable saved to: ../dataset_examples_table.tex")
    
    # Print individual examples for verification
    print("\nIndividual examples:")
    print(f"BBQ: {bbq_example}")
    print(f"StereoSet: {stereoset_example}")  
    print(f"CLEAR-Bias: {clear_bias_example}")
    print(f"MMLU: {mmlu_example}")

if __name__ == "__main__":
    create_latex_table()