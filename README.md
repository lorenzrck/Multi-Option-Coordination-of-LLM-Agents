# Coordination and Conformity in Multi-Option LLM Systems

The thesis asks whether populations of LLM agents still coordinate when more than two options are available, how the measured conformity changes with the number of options, and whether the single-parameter description of the binary framework of De Marzo, Castellano and Garcia carries over to q options. It generalises the binary Curie-Weiss description to the mean-field Potts softmax, simulates focal-agent opinion dynamics for six open-weight models across numbers of opinions and population sizes, measures the models' responses to prepared opinion distributions, and tests how far the measured response accounts for the collective outcomes.

## Contents

```
notebooks/
  01_binary_replication.ipynb       binary Llama-3 70B replication (N = 50, 20 runs)
  02_response_estimation.ipynb      controlled measurements, neutral labels (leader-share slice, structured states,
                                    binary endgame check with labels from the fifty-label pool)
  03_opinion_dynamics.ipynb         multi-option opinion dynamics, 100-sweep horizon
  04_semantic_experiments.ipynb     controlled measurements with meaningful labels, ten opinion sets
  05_calculations.ipynb             count tables from the raw files, Potts fits, bootstrap intervals, RMSE,
                                    beta_eff, structured predictions, semantic fits, endgame fit, quoted numbers
  06_recreate_thesis_figures.ipynb  the 21 thesis figures from the tables below
  07_abliterated_experiments.ipynb  Qwen 2.5 7B against its abliterated derivative: responses and dynamics (Appendix C)
data/plot_sources/                  compact tables (see "Data")
figures/                            output of notebook 06
Coordination-and-Conformity-in-Multi-Option-LLM-Systems.pdf   the thesis
requirements.txt
```

## Setup

```
pip install -r requirements.txt
```

`numpy`, `pandas`, `scipy` and `matplotlib` for notebooks 05 and 06. The model runs (notebooks 01 to 04 and 07) need `vllm` and `torch` with a CUDA GPU

Two Python environments (Jupyter kernels) were used, because Gemma 4 was not supported by a released vLLM at the time and needed a nightly build:

| Kernel | Models | Python | vLLM | torch | transformers |
|---|---|---|---|---|---|
| `Potts (vLLM)`, plain venv + pip | Qwen 2.5 7B/32B, Llama 3.1 8B, Llama 3 70B AWQ | 3.11.6 | 0.11.2 | 2.9.0+cu128 | 4.57.6 |
| `gemma4-vllm`, uv + CUDA 12.9 nightly wheel index | Gemma 4 E4B, Gemma 4 31B | 3.11.6 | 0.23.1rc1.dev16+gb997071ec (nightly) | 2.11.0+cu129 | 5.12.0 |

The binary replication (notebook 01) ran in a third, plain kernel with vllm 0.11.2, torch 2.9.0, numpy 1.26.4 and pandas 2.2.2. 

## Data

`data/plot_sources/` holds compact, plot- and fit-ready tables

| File | Content |
|---|---|
| `binary_replication.csv` | coordination level per step of the 20 binary replication runs |
| `dynamics_summary.csv` | one row per simulation run (443 runs): condition, seed, consensus time or censoring, final state, response counts |
| `dynamics_trajectories.csv` | leader share and number of active opinions ten times per sweep for every run |
| `neutral_response_counts.csv` | controlled measurements with neutral labels: number of replies per state and chosen role |
| `binary_anchor_counts.csv` | the q = 2, N = 50 blocks by displayed label, for the binary validation figure |
| `structured_response_counts.csv` | replies at the eight structured states (q = 5 and 10, N = 50) |
| `endgame_same_label_test.csv` | Qwen 2.5 7B, N = 100, two active opinions: leader adoption at three states with the fixed pair aa/ab and with labels from the fifty-label pool (300 queries each) |
| `semantic_response_counts.csv` | controlled measurements with meaningful labels, by state, leading option and chosen option |
| `neutral_parameters.csv` | fitted conformity strength per block with bootstrap interval, flag for poorly determined estimates, beta_c and RMSE |
| `effective_parameters.csv` | effective conformity per measured state |
| `structured_validation.csv` | observed and predicted leader adoption at the structured states |
| `semantic_parameters.csv`, `semantic_fields.csv` | semantic fits: conformity, overturn lead, preference field per option |
| `abliterated_response_counts.csv` | replies of Qwen 2.5 7B and its abliterated derivative per state |
| `abliterated_dynamics_summary.csv`, `abliterated_dynamics_trajectories.csv` | Qwen 2.5 7B dynamics abliterated and non abliterated  36 runs (2 models, 6 conditions, 3 runs) |

The raw model outputs are not in the repository due to file size limits. Tables in `data/plot_sources/` are exact aggregates of them. Raw files are available on request.

## Attribution

The simulation and measurement code started from the code of the binary study, Giordano De Marzo's repository github.com/giordano-demarzo/LLMs-Opinion-Dynamics (`opinion_dynamics_chatgpt.py`; archived at doi:10.5281/zenodo.17911594). The prompt wording, the random three-character agent names, the joint shuffling of names and opinions, the random assignment of labels and the focal-agent update loop are taken from there and were adapted to batched local inference with vLLM, to q opinions and to the controlled measurement.
