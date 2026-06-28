# NLU Course Project

Transformer used for **language modeling** (LM) and for **joint intent classification and slot
filling** (NLU), each extended with a pretrained-model variant (LoRA-adapted GPT-2 for LM,
fine-tuned BERT/GPT-2 for NLU).


## Structure

```
LM/
  part_A/   from-scratch GPT-2-style decoder LM, trained on Penn Treebank
  part_B/   LoRA adaptation of a pretrained GPT-2 on the same task
NLU/
  part_A/   from-scratch joint intent/slot model, trained on ATIS
  part_B/   fine-tuned pretrained BERT / GPT-2 backbones, joint intent/slot
figures/    plots generated from the experiment logs, used in report.tex
mybib.bib   bibliography for report.tex
```

Each `part_*` directory is self-contained:

- `model.py` — model architecture
- `functions.py` — train/eval loops
- `utils.py` — data loading and preprocessing
- `tracker.py` — experiment logging (`results/experiments.{csv,json}`, training curves, comparison plots)
- `main.py` — entry point; defines and runs experiments
- `results/` — logged metrics, training curves and comparison plots per experiment

## Setup

```bash
conda env create -f nlu_env.yaml
conda activate nlu
```

### Datasets

Datasets are not committed (see `.gitignore`) and must be placed manually:

- **Penn Treebank** (LM): `dataset/PennTreeBank/{ptb.train.txt,ptb.valid.txt,ptb.test.txt}`
- **ATIS** (NLU): `dataset/ATIS/{train.json,test.json}`

Each dataset directory is relative to the corresponding `part_*` folder (e.g.
`LM/part_A/dataset/PennTreeBank/...`).

## Running an experiment

From inside a `part_*` directory:

```bash
python main.py
```

`main.py` defines one or more calls to `run_experiment(...)`; edit the parameters there (or
uncomment previous runs, kept as comments for reference) to try a different configuration.
Each run logs its results to `results/experiments.csv` / `.json`, saves the best/last model
checkpoint under `bin/`, and regenerates the comparison plots in `results/`.

## Results summary

| Task | Best from-scratch | Best pretrained |
|---|---|---|
| LM (test PPL, lower is better) | 33.95 (weight tying) | 20.57 (GPT-2 + LoRA) |
| NLU (test slot F1 / intent acc.) | 0.890 / 0.922 | 0.946 / 0.976 (BERT fine-tuned) |


## Acknowledgments

Claude was used as a coding and writing assistant throughout this project.
