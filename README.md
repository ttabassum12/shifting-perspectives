## Shifting Perspectives: Steering Vector Ensembles for Robust Bias Mitigation in LLMs

Paper: \< arxiv link pending >

This paper presents a novel approach to bias mitigation in large language models (LLMs) by applying steering vectors to modify model activations in forward passes. 

We employ Bayesian optimization to systematically identify effective contrastive pair datasets across nine bias axes. We also introduce Steering Vector Ensembles (SVE), a method that averages multiple individually optimized steering vectors, each targeting a specific bias axis such as age, race, or gender.

The work presents the first systematic investigation of steering vectors for bias mitigation, and we demonstrate that SVE is a powerful and computationally efficient strategy for reducing bias in LLMs, with broader implications for enhancing AI safety.


### Installation and Setup

After you clone this repository:

- Navigate to it using:
  ```bash
  cd shifting-perspectives
  ```
- (Optional but recommended) Create a virtual environment:
  ```bash
  python -m venv venv
  source venv/bin/activate  # On macOS/Linux
  venv\Scripts\activate  # On Windows
  ```
- Install the required dependencies:
  ```bash
  pip install -r requirements.txt
  ```


### Repo Walkthrough

The data folder contains the BBQ dataset, for convenience. The dataset is from the [BBQ repo](https://github.com/nyu-mll/BBQ). Results can be found in [results](<results>).

In the code folder, scripts are numbered in the order they should be run. Scripts [1_bbq_baselines.py](<code/1_bbq_baselines.py>) and [2_mmlu_baselines.py](code/2_mmlu_baselines.py) generate the baseline results, and there are corresponding bash files to run these for all three models. Script [3_steering_optimisation.py](<code/3_steering_optimisation.py>) runs the Bayesian Optimization process for each axis of the BBQ dataset and logs all trials, with the best trial logged again at the end of the file. An example log from a trial can be found in [logs](<logs>). This also has a corresponding bash script to run the script for all three models.

Files [4_evaluation.py](<code/4_evaluation.py>) and [6_avg_improvements.py](<code/6_avg_improvements.py>) simply load results and print tables used in the paper. Script [5_get_full_steered.py](<code/5_get_full_steered.py>) calculates and saves results used in Tables 2, 3 and 4 in the paper. All code was run on an Nvidia RTX 6000 Ada GPU (50GB RAM).

Finally, graphs used in the paper are generated in the two notebooks in the home directory of the repo.

**Note:** You are likely to need a HuggingFace token in a .env file, as access to some of the models used have gated access. An example can be found in [.env.example](.env.example).


### Novel Contributions

- the first application of steering vectors to social biases such as racial, gender, socioeconomic and age biases,
- a framework to systematically identify effective contrastive datasets via Bayesian optimization, enhancing the robustness of previous activation steering methods,
- and Steering Vector Ensembles (SVE), a method for modifying activations in forward passes by combining individually tuned steering vectors.

<!-- ### Citing Shifting Perspectives
If you use Shifting Perspectives in your research, please use the following bib entry to cite the [reference paper](https://aclanthology.org/2024.emnlp-main.1035/).
```
@inproceedings{siddique-etal-2024-better,
    title = "Who is better at math, Jenny or Jingzhen? Uncovering Stereotypes in Large Language Models",
    author = "Siddique, Zara  and Turner, Liam  and Espinosa-Anke, Luis",
    editor = "Al-Onaizan, Yaser  and Bansal, Mohit  and Chen, Yun-Nung",
    booktitle = "Proceedings of the 2024 Conference on Empirical Methods in Natural Language Processing",
    month = nov,
    year = "2024",
    address = "Miami, Florida, USA",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2024.emnlp-main.1035/",
    doi = "10.18653/v1/2024.emnlp-main.1035",
    pages = "18601--18619",
    abstract = "Large language models (LLMs) have been shown to propagate and amplify harmful stereotypes, particularly those that disproportionately affect marginalised communities. To understand the effect of these stereotypes more comprehensively, we introduce GlobalBias, a dataset of 876k sentences incorporating 40 distinct gender-by-ethnicity groups alongside descriptors typically used in bias literature, which enables us to study a broad set of stereotypes from around the world. We use GlobalBias to directly probe a suite of LMs via perplexity, which we use as a proxy to determine how certain stereotypes are represented in the model's internal representations. Following this, we generate character profiles based on given names and evaluate the prevalence of stereotypes in model outputs. We find that the demographic groups associated with various stereotypes remain consistent across model likelihoods and model outputs. Furthermore, larger models consistently display higher levels of stereotypical outputs, even when explicitly instructed not to."
}
``` -->