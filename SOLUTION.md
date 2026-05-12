# Solution implementation 

After following all the implementation from `README.md`:

## Overview

Large (and small) language models sometimes *hallucinate* — they generate
plausible-sounding text that is factually incorrect.  This competition asks you
to build a **lightweight binary classifier** (called a *probe*) that reads the
model's internal hidden states and predicts whether a given response is
truthful (`label = 0`) or hallucinated (`label = 1`).

The language model used throughout is **[Qwen/Qwen2.5-0.5B](https://huggingface.co/Qwen/Qwen2.5-0.5B)** — a
decoder-only causal transformer with 24 layers and a hidden dimension of 896.
It fits comfortably on a free Google Colab T4 GPU.

**Primary ranking metric:** Accuracy on the held-out `test.csv`.

## Repository Structure

```
SMILES-HALLUCINATION-DETECTION/
├── data/
│   ├── dataset.csv        # Labelled training data (prompt, response, label)
│   └── test.csv           # Unlabelled competition test set
│
├── solution.py            # Main script - run to create a 
│
│
├── predictions.csv        # Main label output 
│
│
├── results.json           # Main results
│
├── aggregation.py         # Layer selection, token pooling, geometric features
├── probe.py               # HallucinationProbe — the binary classifier
├── splitting.py           # Train / validation / test split strategy
│
│   ── Fixed infrastructure (do not edit) ───────────────────────────────
├── model.py               # Loads Qwen2.5-0.5B and exposes get_model_and_tokenizer()
├── evaluate.py            # Evaluation loop, metrics, summary table, JSON output
│
├── requirements.txt       # Python dependencies
└── LICENSE
```


## Quick Start

### Google Colab

Open the terminal in Colab and run:

```python
git clone https://github.com/arofenitra/SMILES-HALLUCINATION-DETECTION.git
cd SMILES-HALLUCINATION-DETECTION
pip install -r requirements.txt
python solution.py
```

### Local Setup

```bash
git clone https://github.com/arofenitra/SMILES-HALLUCINATION-DETECTION.git
cd SMILES-HALLUCINATION-DETECTION

python -m venv .venv
source .venv/bin/activate        # Linux / macOS
# .venv\Scripts\activate.bat     # Windows

pip install -r requirements.txt
python solution.py
```

- Final solution description: components I modified
-> `aggregation.py` (using entropy on hidden states by transforming them into a softmax probabilities and using mean average. They are across the token length up to the padding token) and , `probe.py` (improved architecture with a dropout of 0.3, MLP of deeper size), `splittinh.py` to implement 5 folds validations.
- Reason of choices
-> Entropy can detect hallucination, mean for average structures.
- What contributed most to improving the metric : better architecture, adding entropy
- Experiments and failed attempts: hidden layer of 256 only simple architecture, using the last tokens only for the hidden states, using the mean across all tokens.
-  Why they did not work or were discarded : lower roc-auc score, not strong enough for hallucination detection.

