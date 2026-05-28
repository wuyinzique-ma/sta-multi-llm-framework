# STA Multi-LLM Framework

A multi-LLM framework for ontology-based semantic annotation of energy-domain tables, supporting both Column Type Annotation (CTA) and joint CTA+CPA settings. The framework combines multi-model candidate generation with confidence-based soft voting (MSV) for robust semantic prediction.

## Framework Overview

Three experimental settings are supported:

- **EDM** (Ensemble Decision-Making): BFS-based ontology traversal with majority voting, used as baseline
- **MSV Base**: Single-stage candidate generation with raw ontology input and soft voting (CTA-only)
- **MSV Structured**: Single-stage candidate generation with DAG-based ontology input and soft voting (CTA-only)
- **MSV CNP**: Joint CTA+CPA annotation with flattened ontology input and soft voting

## Environment Setup

Tested with `python==3.12.4`.

```bash
conda create -n eswc2025 python=3.12.4
conda activate eswc2025
pip install -r requirements.txt
pip install -e .
```

## API Keys

Fill in your API keys directly in the `main` notebook of each experiment.
For example:

```python
llm1 = BaseChatOpenAI(
    model="deepseek-chat",
    openai_api_key="YOUR_API_KEY",
    ...
)
```

## Running Experiments

Notebooks are located in `run/`. Run them in the following order:

1. `run_*_main.ipynb` — candidate generation
2. `run_*_eval.ipynb` — threshold sweep and evaluation
3. `run_*_threshold_eval.ipynb` — evaluation with pre-filtering

For each experiment, the optimal confidence threshold is selected based on node-level micro F1.

Results are saved to `results/` automatically.
