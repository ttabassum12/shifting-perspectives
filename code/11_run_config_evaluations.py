import sys  
import os
import glob
import pandas as pd
import logging
from datetime import datetime
from dialz import SteeringModel, SteeringVector
from transformers import AutoTokenizer

# Import functions from the individual evaluation files  
# Note: Using importlib because Python module names can't start with numbers
from importlib import import_module

bbq_eval = import_module('6_bbq_evaluation')
mmlu_eval = import_module('7_mmlu_evaluation') 
stereoset_eval = import_module('8_stereoset_evaluation')
crows_eval = import_module('9_crows_pairs_evaluation')
clear_bias_eval = import_module('10_clear_bias_evaluation')

# Global configuration
USE_FAIRNESS_PROMPT = False  # Set to True to enable fairness prompting
USE_SELF_DEBIAS = True    # Set to True to enable self-debiasing

if len(sys.argv) > 1:
    model_name = sys.argv[1]
else:
    raise ValueError("Model name must be provided as a command-line argument.")

model_short_names = {
    "Qwen/Qwen2.5-7B-Instruct": "qwen",
    "meta-llama/Llama-3.1-8B-Instruct": "llama",
    "mistralai/Mistral-7B-Instruct-v0.1": "mistral",
}

model_short_name = model_short_names.get(model_name)
if not model_short_name:
    raise ValueError(f"Unknown model name: {model_name}")

# Initialize tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token_id = tokenizer.eos_token_id



def run_evaluations_for_config(config_file):
    """Run all evaluations for a given config file by calling functions from individual files."""
    config_name = os.path.basename(config_file).replace('.csv', '')
    print(f"\nRunning evaluations for config: {config_name}")
    
    # Load config
    config_df = pd.read_csv(config_file)
    print(f"Loaded {len(config_df)} configurations")
    
    results = []
    
    for _, config_row in config_df.iterrows():
        axis = config_row['axis']
        vector_type = config_row['vector_type']
        layer = int(config_row['layer'])
        coeff = config_row['coeff']
        bbq_accuracy = config_row['bbq_accuracy']
        mmlu_accuracy = config_row['mmlu_accuracy']
        
        print(f"\n  Processing {axis} (layer={layer}, coeff={coeff})...")
        
        # Load model and vector for this configuration
        model = SteeringModel(model_name, [layer])
        model.half()
        vector = SteeringVector.import_gguf(f'../vectors/{model_short_name}/{vector_type}/{axis}.gguf')
        
        # Initialize result row with config data
        result_row = {
            'axis': axis,
            'vector_type': vector_type,
            'layer': layer,
            'coeff': coeff,
            'bbq_accuracy': bbq_accuracy,
            'mmlu_accuracy': mmlu_accuracy,
            'fairness_prompt': USE_FAIRNESS_PROMPT,
            'self_debias': USE_SELF_DEBIAS
        }
        
        # Call evaluation functions with model, vector, and axis
        try:
            print("    Running BBQ evaluation...")
            bbq_result = bbq_eval.run_bbq_evaluation(model, vector, coeff, axis, tokenizer, USE_FAIRNESS_PROMPT, USE_SELF_DEBIAS)
            print("      BBQ evaluation completed")
            # Add BBQ results with prefix
            if bbq_result:
                for key, value in bbq_result.items():
                    if key != 'axis':  # Don't duplicate axis
                        result_row[f'bbq_{key}'] = value
        except Exception as e:
            print(f"      Error in BBQ evaluation: {e}")
        
        try:
            print("    Running MMLU evaluation...")
            mmlu_result = mmlu_eval.run_mmlu_evaluation(model, vector, coeff, axis, tokenizer, USE_FAIRNESS_PROMPT, USE_SELF_DEBIAS)
            print("      MMLU evaluation completed")
            # Add MMLU results with prefix
            if mmlu_result:
                for key, value in mmlu_result.items():
                    if key not in ['axis', 'coeff']:  # Don't duplicate these
                        result_row[f'mmlu_{key}'] = value
        except Exception as e:
            print(f"      Error in MMLU evaluation: {e}")
        
        # Only run bias evaluations for relevant axes
        bias_evaluations = [
            ("StereoSet", stereoset_eval.run_stereoset_evaluation, ['gender', 'race', 'religion'], 'stereoset'),
            #("CrowS-Pairs", crows_eval.run_crows_pairs_evaluation, ['gender', 'race', 'religion', 'age', 'nationality', 'socioeconomic', 'appearance', 'disability'], 'crows'),
            #("Clear Bias", clear_bias_eval.run_clear_bias_evaluation, ['gender', 'race', 'age', 'disability', 'religion', 'socioeconomic'], 'clear_bias')
        ]
        
        for eval_name, eval_func, relevant_axes, prefix in bias_evaluations:
            if axis in relevant_axes:
                try:
                    print(f"    Running {eval_name} evaluation...")
                    eval_result = eval_func(model, vector, coeff, axis, tokenizer, USE_FAIRNESS_PROMPT, USE_SELF_DEBIAS)
                    print(f"      {eval_name} evaluation completed")
                    # Add evaluation results with prefix
                    if eval_result:
                        for key, value in eval_result.items():
                            if key not in ['axis', 'coeff']:  # Don't duplicate these
                                result_row[f'{prefix}_{key}'] = value
                except Exception as e:
                    print(f"      Error in {eval_name} evaluation: {e}")
                    # Add None values for failed evaluations
                    result_row[f'{prefix}_error'] = str(e)
            else:
                print(f"    Skipping {eval_name} evaluation (axis '{axis}' not relevant)")
                # Add None values for skipped evaluations
                result_row[f'{prefix}_skipped'] = True
        
        results.append(result_row)
    
    # Save results to CSV
    results_df = pd.DataFrame(results)
    os.makedirs(f"../results/{model_short_name}", exist_ok=True)
    results_file = f"../results/{model_short_name}/{config_name}.csv"
    results_df.to_csv(results_file, index=False)
    
    print(f"\nAll evaluations complete for config: {config_name}")
    print(f"Results saved to {results_file}")
    print(f"Saved {len(results)} rows with {len(results_df.columns)} columns")


def setup_logging():
    """Set up logging to redirect all output to a log file."""
    # Create timestamp for unique log file name
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = f"../logs/evaluation_run_{timestamp}.log"
    
    # Set up logging configuration
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(sys.stdout)  # Also show on console
        ]
    )
    
    # Redirect print statements to logging
    class PrintToLog:
        def write(self, text):
            if text.strip():  # Only log non-empty lines
                logging.info(text.strip())
        def flush(self):
            pass
    
    sys.stdout = PrintToLog()
    sys.stderr = PrintToLog()
    
    return log_file


def main():
    """Run evaluations for baseline config file only."""
    log_file = setup_logging()
    print(f"Logging to: {log_file}")
    print(f"Fairness prompting enabled: {USE_FAIRNESS_PROMPT}")
    print(f"Self-debiasing enabled: {USE_SELF_DEBIAS}")
    
    # Only run on baseline config file
    baseline_config = "../data/configs/baselines.csv"
    
    if not os.path.exists(baseline_config):
        raise FileNotFoundError(f"Baseline config file not found: {baseline_config}")
    
    print(f"Running evaluations for baseline config only: {baseline_config}")
    run_evaluations_for_config(baseline_config)
    
    print("\nBaseline evaluations complete!")


if __name__ == "__main__":
    main()