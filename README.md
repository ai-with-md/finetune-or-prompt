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

## Data

[Banking77](https://huggingface.co/datasets/PolyAI/banking77) (Casanueva et al., 2020), published by PolyAI under CC BY 4.0: 10,003 training and 3,080 test messages, 77 intents. The data is downloaded by a script at a pinned revision and checked against SHA-256 checksums; it is not stored in this repo.

## Run it

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
python scripts/download_data.py
pytest -q
```

## License

Code: MIT. Banking77: CC BY 4.0, by PolyAI.
