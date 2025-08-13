## Shifting Perspectives: Steering Vectors for Robust Bias Mitigation in LLMs

Paper: https://arxiv.org/abs/2503.05371

We present a novel approach to bias mitigation in large language models (LLMs) by applying steering vectors to modify model activations in forward passes. 

We compute 8 steering vectors, each corresponding to a different social bias axis, such as age, gender, or race, on a training subset of the BBQ dataset and compare the effectiveness of these to 3 additional bias mitigation methods across 4 datasets. When optimized on the BBQ dataset, our individually tuned steering vectors achieve average improvements of 12.8% on BBQ, 8.3% on CLEAR-Bias, and 1% on StereoSet, and show improvements over prompting and Self-Debias in all cases, and improvements over fine-tuning in 12 out of 17 evaluations. 

In addition, steering vectors showed the lowest impact on MMLU scores of the four bias mitigation methods tested. The work presents the first systematic investigation of steering vectors for bias mitigation, and we demonstrate that they are a powerful and computationally efficient strategy for reducing bias in LLMs, with broader implications for enhancing AI safety.


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

The *raw_data* folder contains the BBQ dataset, for convenience. The dataset is from the [BBQ repo](https://github.com/nyu-mll/BBQ). Results can be found in [results](<results>).

In the code folder, scripts are numbered in the order they should be run. All code was run on an Nvidia RTX 6000 Ada GPU (50GB RAM).


**Note:** You are likely to need a HuggingFace token in a .env file, as access to some of the models used have gated access. An example can be found in [.env.example](.env.example).


### Novel Contributions

- the first application of steering vectors to social biases such as racial, gender, socioeconomic and age biases,
- comprehensive empirical results comparing steering vectors to no intervention, prompting, fine‑tuning, and Self‑Debias, showing superior bias reduction on BBQ, CLEAR‑Bias, StereoSet, and MMLU with minimal impact on overall performance,
- and demonstration that steering vectors trained on one bias‐specific dataset transfer effectively to other tasks and models, underscoring their robustness and practicality.

### Citing Shifting Perspectives
If you use Shifting Perspectives in your research, please use the following bib entry to cite the [reference paper](https://arxiv.org/abs/2503.05371).
```
@misc{siddique2025shiftingperspectives,
      title={Shifting Perspectives: Steering Vectors for Robust Bias Mitigation in LLMs}, 
      author={Zara Siddique and Irtaza Khalid and Liam D. Turner and Luis Espinosa-Anke},
      year={2025},
      eprint={2503.05371},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2503.05371}, 
}
```