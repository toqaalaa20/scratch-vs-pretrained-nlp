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
```

Each `part_*` directory is self-contained:

- `model.py` — model architecture
- `functions.py` — train/eval loops and result reporting (plots, metric printing)
- `utils.py` — data loading and preprocessing
- `main.py` — entry point; defines and runs the experiment
- `dataset/` — dataset files (not committed, see below)
- `bin/` — best model checkpoint
- `results/` — training curves from past hyperparameter-search runs

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

`main.py` calls `run_experiment(...)` with the best configuration found (earlier configurations
explored during hyperparameter search are kept as comments for reference). Each run prints the
final metrics, saves the best model checkpoint under `bin/`, and writes a training-curve plot to
`results/`.

## Results summary

| Task | Best from-scratch | Best pretrained |
|---|---|---|
| LM (test PPL, lower is better) | 33.95 (weight tying) | 20.57 (GPT-2 + LoRA) |
| NLU (test slot F1 / intent acc.) | 0.890 / 0.922 | 0.946 / 0.976 (BERT fine-tuned) |


## Acknowledgments

Claude was used as a coding and writing assistant throughout this project.
