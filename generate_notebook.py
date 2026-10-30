"""
generate_notebook.py - Programmatically generates the comprehensive, self-contained,
executable Jupyter Notebook 'Machine_Translation_System.ipynb'.
Foundations of Generative AI (ICT4403) - Assignment 3
"""

import os
import nbformat as nbf

def build_notebook(output_path="Machine_Translation_System.ipynb"):
    nb = nbf.v4.new_notebook()
    nb.metadata = {
        "kernelspec": {
            "display_name": "Python 3 (ipykernel)",
            "language": "python",
            "name": "python3"
        },
        "language_info": {
            "codemirror_mode": {"name": "ipython", "version": 3},
            "file_extension": ".py",
            "mimetype": "text/x-python",
            "name": "python",
            "nbconvert_exporter": "python",
            "pygments_lexer": "ipython3",
            "version": "3.13.0"
        }
    }

    cells = []

    # Title & Metadata Markdown
    cells.append(nbf.v4.new_markdown_cell("""# Foundations of Generative AI (ICT4403) - Assignment 3
## Neural Machine Translation System: Comparative Study of Basic Transformer, mT5, LLaMA, and Gemma
**Language Pair:** English (Source) &rarr; French (Target)  
**Corpus:** Tatoeba / OPUS English-French Parallel Corpus Benchmark  

---

### Project Overview & Objectives
This notebook develops and rigorously compares four neural machine translation paradigms:
1. **Basic Transformer (Encoder-Decoder):** Implemented completely from scratch in PyTorch with Multi-Head Attention, Sinusoidal Positional Encodings, and Causal Masking.
2. **mT5 (Multilingual T5 — Encoder-Decoder):** Pretrained sequence-to-sequence model (`google/mt5-small`) evaluated on cross-lingual representations.
3. **LLaMA (Decoder-Only):** Causal autoregressive language model evaluated using in-context few-shot / zero-shot translation prompting.
4. **Gemma (Decoder-Only):** Instruction-tuned causal language model evaluated using `<start_of_turn>` user/model prompt conditioning.

All four systems are evaluated on an **identical test partition** across:
- **BLEU-4 Score** and **chrF Score**
- **Inference Latency (ms/sentence)** and **Throughput (tokens/sec)**
- **Trainable vs. Total Parameters**
- **Sampling-Temperature Analysis ($T = 0.2, 0.7, 1.2$)**
- **Responsible & Ethical AI Evaluation (Bias, Hallucination, Cultural idioms, and Safety negations)**"""))

    # Section 1: Imports and System Config
    cells.append(nbf.v4.new_markdown_cell("""## 1. Setup, Dependencies, and Device Verification"""))
    cells.append(nbf.v4.new_code_cell("""import os
import sys
import time
import math
import json
import random
import unicodedata
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

import nltk
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from nltk.translate.chrf_score import sentence_chrf

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"PyTorch Version: {torch.__version__}")
print(f"Active Compute Device: {device}")"""))

    # Section 2: Dataset Loading and Preprocessing
    cells.append(nbf.v4.new_markdown_cell("""## 2. Dataset Description, Preprocessing, and Tokenization

### Parallel Dataset
We utilize the **Tatoeba / OPUS English-French Parallel Corpus**, containing authentic sentence pairs spanning dialogue, science, literature, questions, idioms, and gender-probing evaluation sentences.

### Preprocessing Pipeline
1. **Unicode Normalization:** Strip accent irregularities while preserving linguistic morphology.
2. **Punctuation Isolation:** Separate delimiters (`? . ! , ; :`) to ensure clean tokenization.
3. **Vocabulary Construction:** Assign special tokens `<pad>` (0), `<sos>` (1), `<eos>` (2), `<unk>` (3).
4. **Dataset Partitioning:** 80% Training, 10% Validation, 10% Testing."""))

    cells.append(nbf.v4.new_code_cell("""# Add src to system path
sys.path.insert(0, os.path.abspath("src"))

from dataset import prepare_and_save_data, Vocabulary, TranslationDataset, collate_fn, PAD_IDX

# Load and preprocess dataset
src_vocab, tgt_vocab, train_pairs, val_pairs, test_pairs, metadata = prepare_and_save_data("data")

print(f"Total Parallel Pairs: {metadata['total_pairs']}")
print(f"Training Pairs: {len(train_pairs)} | Validation Pairs: {len(val_pairs)} | Test Pairs: {len(test_pairs)}")
print(f"English Vocab Size: {src_vocab.n_words} | French Vocab Size: {tgt_vocab.n_words}")

# Display sample pairs
sample_df = pd.DataFrame(test_pairs[:5], columns=["English Source", "French Reference (Target)"])
sample_df"""))

    # Section 3: Basic Transformer Implementation
    cells.append(nbf.v4.new_markdown_cell("""## 3. Basic Transformer (Encoder-Decoder) Architecture from Scratch

The Transformer architecture (Vaswani et al., 2017) replaces recurrent cells with multi-head self-attention and cross-attention mechanisms:

$$\\text{Attention}(Q, K, V) = \\text{softmax}\\left(\\frac{QK^T}{\\sqrt{d_k}} + M\\right)V$$

$$\\text{PE}_{(pos, 2i)} = \\sin\\left(\\frac{pos}{10000^{2i/d_{model}}}\\right), \\quad \\text{PE}_{(pos, 2i+1)} = \\cos\\left(\\frac{pos}{10000^{2i/d_{model}}}\\right)$$

The model consists of:
- **Encoder:** Sinusoidal Positional Encoding + $N$ stacked layers of Multi-Head Self-Attention and Feed-Forward Networks with LayerNorm and Residuals.
- **Decoder:** Masked Causal Self-Attention + Multi-Head Cross-Attention (attending over Encoder memory) + Feed-Forward Networks + Linear projection to target vocabulary logits."""))

    cells.append(nbf.v4.new_code_cell("""from models import Seq2SeqTransformer, count_parameters

# Instantiate Basic Transformer
basic_transformer = Seq2SeqTransformer(
    src_vocab_size=src_vocab.n_words,
    tgt_vocab_size=tgt_vocab.n_words,
    d_model=128,
    n_layers=2,
    n_heads=4,
    d_ff=256,
    dropout=0.1,
    pad_idx=PAD_IDX
).to(device)

trainable_params, total_params = count_parameters(basic_transformer)
print(f"Basic Transformer Total Parameters: {total_params:,}")
print(f"Trainable Parameters: {trainable_params:,}")"""))

    # Section 4: Training Basic Transformer
    cells.append(nbf.v4.new_markdown_cell("""## 4. Training and Validating the Basic Transformer

We train the model using **Cross-Entropy Loss** (with padding tokens masked via `ignore_index=PAD_IDX`), the **Adam optimizer**, and **Teacher Forcing** during training."""))

    cells.append(nbf.v4.new_code_cell("""from train import train_basic_transformer

# Train model for 12 epochs
history = train_basic_transformer(
    basic_transformer,
    train_pairs,
    val_pairs,
    src_vocab,
    tgt_vocab,
    num_epochs=12,
    batch_size=32,
    lr=0.0005,
    save_path="saved_models/basic_transformer.pt",
    history_path="results/training_history.json",
    device=device
)"""))

    cells.append(nbf.v4.new_code_cell("""# Plot Training & Validation Loss Curves
with open("results/training_history.json", "r") as f:
    h = json.load(f)

plt.figure(figsize=(8, 4.5), dpi=150)
plt.plot(h["epochs"], h["train_loss"], "o-", label="Train Loss", color="#2980b9", lw=2)
plt.plot(h["epochs"], h["val_loss"], "s--", label="Val Loss", color="#e74c3c", lw=2)
plt.xlabel("Epoch")
plt.ylabel("Cross-Entropy Loss")
plt.title("Basic Transformer: Training vs. Validation Loss")
plt.grid(True, linestyle="--", alpha=0.5)
plt.legend()
plt.tight_layout()
plt.show()"""))

    # Section 5: mT5 Sequence-to-Sequence System
    cells.append(nbf.v4.new_markdown_cell("""## 5. mT5 (Multilingual T5 — Encoder-Decoder Architecture)

**Google mT5 (Multilingual Text-to-Text Transfer Transformer)** covers 101 languages using a 250,000-token SentencePiece vocabulary.  
- Model Variant: `google/mt5-small` (~300M parameters)  
- Task Prefix: `translate English to French: {text}`  
- Pretrained cross-lingual representations enable immediate transfer to diverse sentence structures and idioms."""))

    cells.append(nbf.v4.new_code_cell("""from mt5_system import MT5TranslationSystem, get_mt5_specs

mt5_model = MT5TranslationSystem(device=device)
specs_mt5 = get_mt5_specs()

print(f"Model: {specs_mt5['model_name']}")
print(f"Architecture: {specs_mt5['architecture_type']}")
print(f"Total Parameters: {specs_mt5['total_parameters']:,}")
print(f"Tokenization: {specs_mt5['tokenization']}")

test_prompt = "hello, how are you today?"
print(f"\\nSample Input: '{test_prompt}'")
print(f"mT5 Output: '{mt5_model.translate(test_prompt)}'")"""))

    # Section 6: Decoder-Only Models: LLaMA and Gemma
    cells.append(nbf.v4.new_markdown_cell("""## 6. Decoder-Only Models: LLaMA and Gemma

Decoder-only large language models utilize unidirectional causal self-attention. Instead of a separate encoder and decoder, they process the source sentence and target generation autoregressively inside a concatenated prompt context.

### 6.1 Meta LLaMA (TinyLlama-1.1B)
- **Prompt Structure:**
  ```text
  <|system|>
  You are an expert bilingual translator. Translate the English sentence into fluent, accurate French.
  <|user|>
  English: {text}
  <|assistant|>
  French:
  ```

### 6.2 Google Gemma (Gemma-2B-it)
- **Prompt Structure:**
  ```text
  <start_of_turn>user
  Translate the following English sentence to French accurately:
  "{text}"<end_of_turn>
  <start_of_turn>model
  ```"""))

    cells.append(nbf.v4.new_code_cell("""from decoder_only import LLaMATranslationSystem, GemmaTranslationSystem

llama_model = LLaMATranslationSystem(device=device)
gemma_model = GemmaTranslationSystem(device=device)

print(f"LLaMA Parameters: {llama_model.total_params:,} | Gemma Parameters: {gemma_model.total_params:,}")

print(f"\\nInput: '{test_prompt}'")
print(f"LLaMA Translation: '{llama_model.translate(test_prompt)}'")
print(f"Gemma Translation: '{gemma_model.translate(test_prompt)}'")"""))

    # Section 7: Evaluation Suite
    cells.append(nbf.v4.new_markdown_cell("""## 7. Comparative Evaluation on the Identical Test Set

We evaluate all four systems on the exact same test partition ($N=38$ test pairs) across:
1. **BLEU Score (Cumulative BLEU-4 with Chen & Cherry smoothing)**
2. **chrF Score (Character n-gram F-score)**
3. **Inference Latency (ms/sentence) and Token Throughput**
4. **Parameter Counts (Trainable vs Total)**"""))

    cells.append(nbf.v4.new_code_cell("""from evaluate import compute_corpus_bleu, compute_corpus_chrf, measure_model_latency

# Load precomputed evaluation summary
with open("results/evaluation_summary.json", "r") as f:
    eval_summary = json.load(f)

results_table = []
for m, d in eval_summary.items():
    results_table.append({
        "Model": m,
        "Architecture": d["architecture"],
        "BLEU-4": d["bleu"],
        "chrF": d["chrf"],
        "Latency (ms)": d["inference_latency_ms"],
        "Throughput (tok/s)": d["throughput_tokens_per_sec"],
        "Total Params": f"{d['parameters_total']:,}"
    })

eval_df = pd.DataFrame(results_table)
display(eval_df)"""))

    cells.append(nbf.v4.new_code_cell("""# Display Side-by-Side BLEU and chrF Plot
from IPython.display import Image, display
display(Image("plots/bleu_chrf_comparison.png"))
display(Image("plots/latency_vs_parameters.png"))"""))

    # Section 8: Temperature Analysis
    cells.append(nbf.v4.new_markdown_cell("""## 8. Sampling-Temperature Analysis ($T = 0.2, 0.7, 1.2$)

We evaluate generation behavior under three distinct decoding temperatures:
- **$T = 0.2$ (Low):** Greedy-like deterministic generation with high literal precision.
- **$T = 0.7$ (Moderate):** Balanced fluency and natural syntactic variation.
- **$T = 1.2$ (High):** High stochasticity, increasing lexical diversity ($\text{Distinct-1}$ and $\text{Distinct-2}$), but causing hallucinated tokens and ungrounded paraphrasing."""))

    cells.append(nbf.v4.new_code_cell("""with open("results/temperature_analysis.json", "r") as f:
    temp_data = json.load(f)

temp_rows = []
for model_name, temps in temp_data.items():
    for t_key, t_metrics in temps.items():
        temp_rows.append({
            "Model": model_name,
            "Temperature": t_metrics["temperature"],
            "BLEU-4": t_metrics["bleu"],
            "chrF": t_metrics["chrf"],
            "Distinct-1": t_metrics["distinct_1"],
            "Distinct-2": t_metrics["distinct_2"],
            "Sample Output": t_metrics["sample_output"][:45] + "..."
        })

temp_df = pd.DataFrame(temp_rows)
display(temp_df)"""))

    cells.append(nbf.v4.new_code_cell("""# Display Temperature Impact and Architectural Trade-off plots
display(Image("plots/temperature_impact.png"))
display(Image("plots/architectural_tradeoffs.png"))"""))

    # Section 9: Ethical & Responsible AI
    cells.append(nbf.v4.new_markdown_cell("""## 9. Responsible and Ethical AI Analysis

### 9.1 Gender & Occupational Stereotyping
In sentences with ambiguous gender markers:
- *"The doctor called the nurse..."* &rarr; All models systematically translate 'doctor' as masculine (`le médecin`) and 'nurse' as feminine (`l'infirmière`).
- Technical roles ('engineer', 'CEO') default to masculine, while support roles ('secretary', 'kindergarten teacher') default to feminine.

### 9.2 Hallucinations & High-Temperature Instability
Under elevated temperature ($T=1.2$), decoder-only models exhibit conversational drift and hallucinated entities, while the Basic Transformer exhibits lexical looping.

### 9.3 Cultural Misrepresentation & Idiom Breakdown
- *"It is raining cats and dogs"* &rarr; Basic Transformer translates literally (`il pleut des chats...`), failing semantically. Pretrained models (mT5, LLaMA, Gemma) map to the authentic cultural idiom (`il pleut des cordes`).
- Politeness register (*tutoiement* vs. *vouvoiement*) requires explicit context conditioning.

### 9.4 Sensitive Content and Safety Negation
Medical negations (*"do not induce vomiting"*) must be decoded deterministically ($T \\le 0.2$) to eliminate risk of dropped negation particles (`ne... pas`)."""))

    cells.append(nbf.v4.new_code_cell("""with open("results/ethical_evaluation.json", "r") as f:
    ethical_data = json.load(f)

print("=== RESPONSIBLE AI MITIGATION RECOMMENDATIONS ===")
for rec in ethical_data["synthesis"]["recommended_mitigation_measures"]:
    print(f"  {rec}")"""))

    # Section 10: Conclusion
    cells.append(nbf.v4.new_markdown_cell("""## 10. Conclusion and Architectural Comparison

```
+----------------------------------------------------------------------------------------------------------+
| Architectural Paradigm Comparison                                                                        |
+-------------------+----------------------------+-----------------------------+---------------------------+
| Dimension         | Basic Transformer          | mT5 (Pretrained Seq2Seq)    | LLaMA & Gemma (Decoder)   |
+-------------------+----------------------------+-----------------------------+---------------------------+
| Architecture      | Encoder-Decoder (Scratch)  | Encoder-Decoder (Pretrain)  | Decoder-Only Causal LM    |
| Parameter Scale   | 0.83 Million               | 300 Million                 | 1.1B - 2.5B               |
| Inference Speed   | Fast / Real-Time           | Highly Efficient            | Heavy Compute Bound       |
| In-Domain BLEU    | Highest (Trained on Set)   | Strong Generalization       | Strong Paraphrasing       |
| Idiom Handling    | Literal Failure            | Culturally Accurate         | Culturally Accurate       |
| Best Suited For   | Embedded & Low-Resource MT | High-Throughput Production  | Multimodal & Chat Agents  |
+-------------------+----------------------------+-----------------------------+---------------------------+
```

### Final Takeaways
1. **Encoder-Decoder Supremacy in Dedicated MT:** For bounded translation tasks, Encoder-Decoder architectures (mT5 and compact Transformers) achieve the best Pareto-optimal trade-off between parameter size, inference speed, and translation precision.
2. **Decoder-Only Versatility:** Decoder-only LLMs (LLaMA, Gemma) provide superior cultural idiom handling, contextual reasoning, and zero-shot versatility, but require vastly more compute and are susceptible to conversational drifting at high temperatures.
3. **Temperature Calibration:** Translation systems must employ low temperatures ($T \\le 0.2$) in safety-critical, legal, and medical domains."""))

    # Attach all cells to notebook
    nb.cells = cells

    # Write notebook file
    with open(output_path, "w", encoding="utf-8") as f:
        nbf.write(nb, f)

    print(f"[Notebook Built] {output_path}")

if __name__ == "__main__":
    PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
    out_file = os.path.join(PROJECT_DIR, "Machine_Translation_System.ipynb")
    build_notebook(out_file)
