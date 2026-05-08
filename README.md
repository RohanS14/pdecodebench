`data/`: Datasets and descriptions of data. See <a href='https://github.com/RohanS14/pdecodebench/blob/main/data/descriptions/data_spec.txt'>data_spec.txt</a>

`datagen/`: Scripts for dataset corruption and augmentation.

`eval/`: Evaluation pipeline for model outputs.
- `run_eval.py` — Experiment 1: free-generation accuracy across 10 LLMs
- `run_mc_eval.py` — Experiment 2: MCQ confidence via logprob extraction (with text-extraction fallback for reasoning models)
- `prepare_var_probes.py` / `run_var_logprob.py` — Experiment 3: variable log-probability evolution (Appendix A.5)
- `frontier/` — Experiment 4: belief revision with execution summaries (Gemini-2.5-Flash)

`probe/`: Probing experiments on model hidden states (Experiment 5).

`results/`: Eval outputs and model responses for experiments 1 and 2.

`viz/`: Visualization scripts.
- `paper_figures.py` — generates static figures for the paper
- `visualize_v3.py` / `visualize_v4_enhanced.py` — interactive dashboards for experiments 1 and 2
- `visualize_var_logprob.py` — interactive dashboard for experiment 3
