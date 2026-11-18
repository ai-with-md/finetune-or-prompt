# Fine-tune or prompt?

**When does fine-tuning a small model pay off, compared with prompting a large hosted model?** A study on one public task, answered in quality, cost per 1,000 requests, speed, and the monthly volume where the answer flips.

[![tests](https://github.com/ai-with-md/finetune-or-prompt/actions/workflows/tests.yml/badge.svg)](https://github.com/ai-with-md/finetune-or-prompt/actions/workflows/tests.yml)
![Python 3.9+](https://img.shields.io/badge/python-3.9%2B-blue)
![License: MIT](https://img.shields.io/badge/license-MIT-green)

> Work in progress. Findings will appear here once they are measured.

## The question

A team needs to sort customer messages into 77 banking intents. They can:

1. **Prompt a large hosted model.** No training, pay per request.
2. **Fine-tune a small open model** (150M parameters) on labelled examples and serve it themselves. Pay once for labels and training, then a fixed amount for a server.
3. **Use a small model without fine-tuning** (frozen sentence embeddings plus a linear classifier), the cheapest option of all.

This study measures all three on the same test messages and turns the results into a decision: how many requests a month, how many labelled examples and what price per wrong answer make fine-tuning worth it.

## Notebooks

| # | Question | Status |
|---|---|---|
| [01](notebooks/01_task_and_baseline.ipynb) | What does the task look like, and how far does a model with no fine-tuning get? | Done |
| 02 | Does a fine-tuned small model beat a prompted large one on the same messages? | Next |
| 03 | How many labelled examples does fine-tuning need to catch up? | Planned |
| 04 | What does each option cost per 1,000 requests, and how fast is it? | Planned |
| 05 | At what monthly volume does fine-tuning pay off, and when does the answer flip? | Planned |

## How the study is run

- **Same test messages for every approach**, so differences are measured in pairs, with 95% intervals from resampling the test set.
- **No tuning on the test set.** A validation set is held out from training, and each approach picks one setting on it that is then used at every training size. Tuning per size would flatter the small sizes.
- **Training sizes are nested** (the 10 examples per intent include the 5), with three random draws each, so the learning curve shows the value of more labels rather than the luck of a draw.
- **Models run locally** in a separate environment ([requirements-train.txt](requirements-train.txt)) and save only their test predictions to [results/](results/). The notebooks read those files, so they rerun without model weights or an API key.

## Data

[Banking77](https://huggingface.co/datasets/PolyAI/banking77) (Casanueva et al., 2020), published by PolyAI under CC BY 4.0: 10,003 training and 3,080 test messages, 77 intents. The data is downloaded by a script at a pinned revision and checked against SHA-256 checksums; it is not stored in this repo.

## Run it

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
python scripts/download_data.py
pytest -q
```

To rerun the models themselves (Python 3.12; saved predictions are already in `results/`):

```bash
python3.12 -m venv .venv-train
.venv-train/bin/pip install -r requirements-train.txt -e .
.venv-train/bin/python scripts/frozen_baseline.py   # about a minute on an Apple M4
.venv-train/bin/python scripts/finetune.py          # about 2.5 hours on an Apple M4 GPU; resumes if stopped
```

The hosted models need an Anthropic API key in a `.env` file in the repo root (`ANTHROPIC_API_KEY=...`; the file is ignored by git). Every answer is saved as it arrives, so a stopped run resumes without paying twice, and spending stops before a $15 cap:

```bash
.venv-train/bin/python scripts/run_api.py --dry-run   # the plan and its cost, no paid calls
.venv-train/bin/python scripts/run_api.py --pilot     # 50 messages, checks the prompt cache works
.venv-train/bin/python scripts/run_api.py             # all six runs, about $3
```

## License

Code: MIT. Banking77: CC BY 4.0, by PolyAI.
