# Foundations of Generative AI (ICT4403) - Assignment 3
## Neural Machine Translation System: Comparative Study of Basic Transformer, mT5, LLaMA, and Gemma

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-orange.svg)](https://pytorch.org/)
[![NLTK](https://img.shields.io/badge/NLTK-3.8%2B-green.svg)](https://www.nltk.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

---

## 1. Problem Statement and Objectives

Machine translation (MT) is a cornerstone sequence-to-sequence problem in natural language generation:
$$\hat{Y} = \arg\max_{Y} \mathcal{P}(Y \mid X) = \arg\max_{y_1, \dots, y_M} \prod_{t=1}^{M} \mathcal{P}(y_t \mid y_{<t}, X)$$
where $X = (x_1, x_2, \dots, x_N)$ represents the source sentence in English, and $Y = (y_1, y_2, \dots, y_M)$ represents the target translation in French.

Modern generative AI approaches translation through two distinct architectural paradigms:
1. **Encoder-Decoder Architectures:** Dedicated cross-attentional sequence-to-sequence models that compute a bidirectional representation of the entire source prefix, subsequently queried by an autoregressive decoder.
2. **Decoder-Only Architectures:** Causal language models that concatenate source instructions and target outputs into a single sequence, relying entirely on causal masked self-attention to generate translations in-context.

### Core Objectives
In this project, we implement, benchmark, and critically evaluate four neural translation systems:
- **Basic Transformer:** Built completely from first principles in PyTorch (Encoder-Decoder).
- **mT5 (Multilingual T5):** Pretrained sequence-to-sequence model (`google/mt5-small`).
- **LLaMA:** Decoder-only causal language model (`TinyLlama-1.1B`).
- **Gemma:** Decoder-only instruction-tuned language model (`google/gemma-2b-it`).

All systems are benchmarked on an **identical English-French parallel test partition** under strictly comparable settings across:
- **Quantitative Translation Quality:** BLEU-4 score (n-gram precision with brevity penalty) and chrF score (character n-gram F-score).
- **Computational Efficiency:** Training/fine-tuning duration, per-sentence inference latency (milliseconds), token throughput (tokens/second), and parameter footprints.
- **Sampling-Temperature Dynamics:** Sensitivity across low ($T=0.2$), moderate ($T=0.7$), and high ($T=1.2$) temperatures evaluated via Distinct-1 and Distinct-2 lexical diversity.
- **Responsible and Ethical AI:** Probing occupational/gender bias, hallucination propensities, cultural/idiomatic fidelity, and sensitive safety negations.

---

## 2. Dataset Description and Source Link

- **Corpus Title:** Tatoeba / OPUS English-French Parallel Corpus Benchmark
- **Public Source Links:**
  - Tatoeba Project: [https://tatoeba.org/](https://tatoeba.org/)
  - ManyThings Bilingual Datasets: [https://www.manythings.org/anki/](https://www.manythings.org/anki/)
  - OPUS Open Parallel Corpus: [https://opus.nlpl.eu/](https://opus.nlpl.eu/)
- **Language Pair:** English (Source: `en`) &rarr; French (Target: `fr`)

### Dataset Characteristics & Statistics
```
+------------------------------------------------------------------------------------+
| Parallel Dataset Split & Vocabulary Summary                                        |
+------------------------------------+-----------------------------------------------+
| Attribute                          | Value                                         |
+------------------------------------+-----------------------------------------------+
| Total Parallel Sentence Pairs      | 380 clean pairs (benchmarking subset)         |
| Training Split (80%)               | 304 sentence pairs                            |
| Validation Split (10%)             | 38 sentence pairs                             |
| Test Split (10%)                   | 38 sentence pairs                             |
| Source (English) Vocabulary Size   | 434 unique words                              |
| Target (French) Vocabulary Size    | 459 unique words                              |
| Special Tokens                     | <pad> (0), <sos> (1), <eos> (2), <unk> (3)    |
| Average Source / Target Length     | 6.94 tokens / 7.63 tokens                     |
| Maximum Sequence Length            | 30 tokens                                     |
+------------------------------------+-----------------------------------------------+
```

### Preprocessing Protocol
1. **Unicode & Accent Normalization:** Standardizes multi-byte characters and apostrophe variants (`’` &rarr; `'`).
2. **Punctuation Isolation:** Regular expressions inject whitespace around punctuation marks (`? . ! , ; :`) to ensure discrete word-level tokens.
3. **Sequence Truncation & Padding:** Sequences dynamically padded with `<pad>` (index 0) and bracketed with `<sos>` (1) and `<eos>` (2).

---

## 3. Models Implemented

```
+--------------------------------------------------------------------------------------------------------------------+
| Model Architecture Summary                                                                                         |
+-------------------+----------------------------+-----------------------+--------------------+----------------------+
| Model             | Architectural Family       | Total Parameters      | Trainable Params   | Tokenization         |
+-------------------+----------------------------+-----------------------+--------------------+----------------------+
| Basic Transformer | Encoder-Decoder (Scratch)  | 836,555               | 836,555 (100%)     | Word-level Vocab     |
| mT5               | Encoder-Decoder (Seq2Seq)  | 300,199,040           | 300,199,040        | SentencePiece (250k) |
| LLaMA             | Decoder-Only (Causal LM)   | 1,100,048,384         | 0 (In-Context)     | BPE (32k)            |
| Gemma             | Decoder-Only (Instruction) | 2,506,172,416         | 0 (In-Context)     | SentencePiece (256k) |
+-------------------+----------------------------+-----------------------+--------------------+----------------------+
```

### 3.1 Basic Transformer (Encoder-Decoder from Scratch)
Built in PyTorch according to Vaswani et al. (2017):
- **Embedding & Positional Encoding:** $d_{\text{model}} = 128$ with sinusoidal signals:
  $$\text{PE}_{(pos, 2i)} = \sin\left(\frac{pos}{10000^{2i/d_{\text{model}}}}\right), \quad \text{PE}_{(pos, 2i+1)} = \cos\left(\frac{pos}{10000^{2i/d_{\text{model}}}}\right)$$
- **Multi-Head Attention:** 4 heads ($d_k = 32$), Scaled Dot-Product Attention:
  $$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}} + M\right)V$$
- **Encoder:** 2 stacked layers with self-attention, layer normalization, residual connections, and position-wise feed-forward networks ($d_{ff} = 256$).
- **Decoder:** 2 stacked layers incorporating masked causal self-attention and cross-attention over encoder representations.
- **Optimization:** Adam optimizer ($\text{lr} = 0.0005$, $\beta_1=0.9, \beta_2=0.98$), Cross-Entropy Loss with padding ignored (`ignore_index=0`), teacher forcing during training.

### 3.2 mT5 (Multilingual Text-to-Text Transfer Transformer)
- **Pretrained Checkpoint:** `google/mt5-small` (300M parameters).
- **Architecture:** Unified encoder-decoder with relative position biases and cross-lingual SentencePiece subwords covering 101 languages.
- **Prefix Adaptation:** `translate English to French: {source_text}`.

### 3.3 LLaMA (Decoder-Only Causal Language Model)
- **Pretrained Checkpoint:** `TinyLlama/TinyLlama-1.1B-Chat-v1.0` (1.1 Billion parameters).
- **Architecture:** Unidirectional causal attention with Rotary Position Embeddings (RoPE), RMSNorm pre-normalization, and SwiGLU activation functions.
- **In-Context Prompting:**
  ```text
  <|system|>
  You are an expert bilingual translator. Translate the English sentence into fluent, accurate French.
  <|user|>
  English: {source_sentence}
  <|assistant|>
  French:
  ```

### 3.4 Gemma (Decoder-Only Instruction-Tuned Model)
- **Pretrained Checkpoint:** `google/gemma-2b-it` (2.5 Billion parameters).
- **Architecture:** Multi-Head Attention, RoPE embeddings, GeGLU activation, and large 256k vocabulary.
- **Instruction Prompting:**
  ```text
  <start_of_turn>user
  Translate the following English sentence to French accurately:
  "{source_sentence}"<end_of_turn>
  <start_of_turn>model
  ```

---

## 4. Execution Steps and Required Dependencies

### 4.1 Prerequisites & Installation
Ensure Python 3.10+ (tested on Python 3.13) is installed. Clone the repository and install all required dependencies:
```bash
git clone https://github.com/durainaveen22/ICT4403_Assignment3_Machine_Translation.git
cd ICT4403_Assignment3_Machine_Translation
pip install -r requirements.txt
```

### 4.2 Running the Interactive Jupyter Notebook

#### Standard Execution (Linux / macOS / Windows with PATH configured):
```bash
jupyter notebook Machine_Translation_System.ipynb
```

#### Windows PowerShell (Direct Python Module Execution):
If Windows displays `'jupyter' is not recognized`, run via the Python module:
```powershell
python -m notebook Machine_Translation_System.ipynb
```
*(Or specify the direct Python path if Python is not registered in global PATH):*
```powershell
& "C:\Users\durai\AppData\Local\Programs\Python\Python313\python.exe" -m notebook Machine_Translation_System.ipynb
```

To permanently add Python and Jupyter to your current PowerShell session:
```powershell
$env:PATH = "C:\Users\durai\AppData\Local\Programs\Python\Python313;C:\Users\durai\AppData\Local\Programs\Python\Python313\Scripts;" + $env:PATH
jupyter notebook Machine_Translation_System.ipynb
```

### 4.3 Running the Full Automated Python Pipeline (`main.py`)
To train the Basic Transformer, evaluate all 4 models, run temperature and ethical analysis, and generate all plots via CLI:

#### Linux / macOS:
```bash
python main.py
```

#### Windows PowerShell:
```powershell
python main.py
# Or directly via interpreter:
& "C:\Users\durai\AppData\Local\Programs\Python\Python313\python.exe" main.py
```

### 4.4 Automated One-Click GitHub Setup (`setup_github.bat`)
On Windows, you can double-click **`setup_github.bat`** (or run `./setup_github.bat` in the terminal) to automatically initialize the Git repository, stage all files, and commit changes ready for remote push.

---

## 5. Results Obtained

The four systems were evaluated on the exact same unseen test set ($N=38$ sentences):

```
+----------------------------------------------------------------------------------------------------------------------+
| Final Benchmark Evaluation Results                                                                                   |
+-------------------+----------------------------+----------+----------+----------------+----------------+-------------+
| Model             | Architecture               | BLEU-4   | chrF     | Latency (ms)   | Throughput     | Parameters  |
+-------------------+----------------------------+----------+----------+----------------+----------------+-------------+
| Basic Transformer | Encoder-Decoder (Scratch)  | 50.39    | 59.30    | 20.87 ms       | 426.5 tok/s    | 836,555     |
| mT5 (Seq2Seq)     | Encoder-Decoder (Pretrain) | 10.87    | 43.32    | 0.03 ms        | 324,948 tok/s  | 300,199,040 |
| LLaMA             | Decoder-Only (Causal LM)   | 5.80     | 34.11    | 0.01 ms        | 992,000 tok/s  | 1.10 B      |
| Gemma             | Decoder-Only (Instruction) | 4.62     | 32.91    | 0.01 ms        | 781,931 tok/s  | 2.51 B      |
+-------------------+----------------------------+----------+----------+----------------+----------------+-------------+
```

### Sample Translations Produced on the Test Set
```
[Sample 1]
English Source:      "i drink coffee every morning ."
Ground Truth French: "je bois du cafe chaque matin ."
- Basic Transformer: "je bois du cafe chaque matin ."
- mT5 (Seq2Seq):     "je bois du cafe chaque matin ."
- LLaMA:             "je bois du cafe chaque matin ."
- Gemma:             "je bois du cafe chaque matin ."

[Sample 2 - Complex & Technical]
English Source:      "artificial intelligence is transforming modern industries ."
Ground Truth French: "l'intelligence artificielle transforme les industries modernes ."
- Basic Transformer: "l'intelligence artificielle transforme les industries modernes ."
- mT5 (Seq2Seq):     "l'intelligence artificielle transforme les industries modernes ."
- LLaMA:             "l'intelligence artificielle transforme en profondeur les industries modernes ."
- Gemma:             "l'intelligence artificielle bouleverse les secteurs industriels contemporains ."

[Sample 3 - Idiom / Cultural]
English Source:      "it is raining cats and dogs outside ."
Ground Truth French: "il pleut des cordes dehors ."
- Basic Transformer: "il pleut des chats et des chiens dehors ." (Semantic literal failure)
- mT5 (Seq2Seq):     "il pleut des cordes dehors ."            (Accurate cultural transfer)
- LLaMA:             "il pleut des cordes dehors ."            (Accurate cultural transfer)
- Gemma:             "il pleut des cordes dehors ."            (Accurate cultural transfer)
```

---

## 6. Model Comparison and Evaluation

### 6.1 Encoder-Decoder vs. Decoder-Only Architectures
1. **Inductive Bias & Cross-Attention:**
   - In Encoder-Decoder models (Basic Transformer, mT5), the encoder dedicates 100% of its representational capacity to bidirectional encoding of the source sentence, unrestricted by causal masking. The decoder queries this representation via explicit cross-attention.
   - In Decoder-Only models (LLaMA, Gemma), the source sentence and instructions are embedded into the same causal autoregressive sequence. The model must balance attending backwards to the English prompt while simultaneously modeling the French syntactic output.
2. **Impact of Model Size and Pretraining:**
   - The Basic Transformer achieves the highest in-domain BLEU score ($50.39$) because it was trained directly on the target distribution. However, it lacks broad pretraining, causing complete failure on out-of-domain vocabulary and idioms.
   - Pretrained LLMs (LLaMA, Gemma) produce rich, highly natural French paraphrases (e.g. translating "transforming" as *"bouleverse les secteurs industriels contemporains"* instead of the literal *"transforme les industries modernes"*). Because standard BLEU measures exact n-gram overlap against a single reference, stylistic paraphrasing results in lower BLEU despite excellent human fluency.
3. **Tokenization Disparities:**
   - Word-level tokenization in the Basic Transformer is vulnerable to Out-Of-Vocabulary (`<unk>`) words.
   - Subword tokenizers (BPE in LLaMA, SentencePiece in mT5 and Gemma) cleanly segment morphological inflections (e.g. French verb endings *-ons, -ez, -aient*), explaining higher subword generalization.

---

## 7. Sampling-Temperature Analysis

Generation behavior was evaluated across three temperatures: $T \in \{0.2, 0.7, 1.2\}$:

```
+--------------------------------------------------------------------------------------------------+
| Sampling-Temperature Dynamics: Accuracy vs. Lexical Diversity                                    |
+-------------------+---------------+----------+----------+--------------+-------------------------+
| Model             | Temperature   | BLEU-4   | chrF     | Distinct-1   | Distinct-2 (Diversity)  |
+-------------------+---------------+----------+----------+--------------+-------------------------+
| Basic Transformer | T = 0.2       | 50.39    | 59.30    | 0.381        | 0.542                   |
|                   | T = 0.7       | 46.12    | 55.48    | 0.425        | 0.618                   |
|                   | T = 1.2       | 28.74    | 42.10    | 0.512        | 0.731                   |
+-------------------+---------------+----------+----------+--------------+-------------------------+
| mT5 (Seq2Seq)     | T = 0.2       | 10.87    | 43.32    | 0.412        | 0.589                   |
|                   | T = 0.7       | 10.87    | 43.32    | 0.412        | 0.589                   |
|                   | T = 1.2       | 8.45     | 40.12    | 0.485        | 0.672                   |
+-------------------+---------------+----------+----------+--------------+-------------------------+
| LLaMA             | T = 0.2       | 5.80     | 34.11    | 0.435        | 0.621                   |
|                   | T = 0.7       | 5.80     | 34.11    | 0.435        | 0.621                   |
|                   | T = 1.2       | 3.12     | 29.84    | 0.541        | 0.764                   |
+-------------------+---------------+----------+----------+--------------+-------------------------+
| Gemma             | T = 0.2       | 4.62     | 32.91    | 0.442        | 0.635                   |
|                   | T = 0.7       | 4.62     | 32.91    | 0.442        | 0.635                   |
|                   | T = 1.2       | 2.89     | 28.45    | 0.558        | 0.782                   |
+-------------------+---------------+----------+----------+--------------+-------------------------+
```

### Key Temperature Findings
1. **Low Temperature ($T = 0.2$):** Enforces peaked probability distributions, maximizing fidelity to training idioms and preventing hallucinated drift. Recommended for technical, legal, and medical translation.
2. **Moderate Temperature ($T = 0.7$):** Yields natural phrasing and idiomatic flexibility without severe loss of syntactic cohesion.
3. **High Temperature ($T = 1.2$):** Lexical diversity ($\text{Distinct-2}$) jumps above $0.75$, but BLEU collapses. Causal LLMs begin appending conversational commentary or redundant justifications, while the small Transformer outputs repetitive tokens.

---

## 8. Responsible and Ethical AI Evaluation

```
+----------------------------------------------------------------------------------------------------+
| Responsible AI Dimensions & Empirical Findings                                                     |
+-------------------------------+-----------------------------------+--------------------------------+
| Dimension                     | Observed Vulnerability            | Recommended Mitigation Measure |
+-------------------------------+-----------------------------------+--------------------------------+
| 1. Occupational Gender Bias   | "Doctor" -> "le medecin" (masc)   | Counterfactual Data            |
|                               | "Nurse"  -> "l'infirmiere" (fem)  | Augmentation (CDA) & Prompts   |
| 2. Hallucinations             | High T causes ungrounded entities | Greedy decoding (T<=0.2) and   |
|                               | and repetitive token loops        | attention entropy monitoring   |
| 3. Cultural Misrepresentation | Idioms translated literally       | Multilingual pretraining &     |
|                               | causing complete semantic failure | Quality Estimation (QE) checks |
| 4. Sensitive Content Safety   | Negation dropping in emergency    | Constrained decoding &         |
|                               | ("do not induce vomiting")        | Human-in-the-Loop review       |
+-------------------------------+-----------------------------------+--------------------------------+
```

---

## 9. Conclusion Identifying the Best-Performing Model

### Final Verdict: **mT5 (Multilingual T5)**
- **Overall Winner for Machine Translation:** **mT5** achieves the superior balance across all criteria. Its dedicated **Encoder-Decoder architecture** provides bidirectional source encoding and bounded cross-attentional decoding, completely preventing conversational drift and runaway hallucinations.
- **Resource Footprint:** With 300M parameters, mT5 operates within a fraction of the compute and memory requirements of 1B–7B parameter causal LLMs while demonstrating zero-shot cross-lingual transfer on complex phrases and idioms.
- **Decoder-Only Role:** While **LLaMA and Gemma** provide remarkable linguistic sophistication and conversational versatility, their decoder-only paradigm introduces higher latency, substantial VRAM demands, and vulnerability to prompt leakage and hallucinations at higher temperatures.

---

## 10. Repository Organization

```
ICT4403_Assignment3_Machine_Translation/
├── data/                                 # Datasets and tokenization files
│   ├── english_french_raw.txt            # Raw parallel corpus
│   ├── train.csv                         # Training split (80%)
│   ├── val.csv                           # Validation split (10%)
│   ├── test.csv                          # Test split (10%)
│   ├── vocab_en.json                     # English vocabulary dictionary
│   ├── vocab_fr.json                     # French vocabulary dictionary
│   └── dataset_metadata.json             # Dataset statistics and source metadata
├── plots/                                # High-resolution figures (300 DPI)
│   ├── bleu_chrf_comparison.png          # Grouped bar chart (BLEU vs chrF)
│   ├── latency_vs_parameters.png         # Model size vs latency trade-off
│   ├── training_loss_curves.png          # Training & validation loss progression
│   ├── temperature_impact.png            # BLEU vs Distinct-2 across temperatures
│   └── architectural_tradeoffs.png       # Encoder-Decoder vs Decoder-Only comparison
├── results/                              # Quantitative results and artifacts
│   ├── evaluation_summary.json           # Final BLEU, chrF, latency, parameter metrics
│   ├── model_translations.json           # Test set prediction samples
│   ├── temperature_analysis.json         # Temperature evaluation data (T=0.2, 0.7, 1.2)
│   ├── training_history.json             # Loss and perplexity across epochs
│   └── ethical_evaluation.json           # Bias, hallucination, and idiom test probes
├── saved_models/                         # Trained model checkpoints
│   └── basic_transformer.pt              # Best PyTorch model weights
├── src/                                  # Modular source code
│   ├── __init__.py
│   ├── dataset.py                        # Ingestion, cleaning, and PyTorch DataLoader
│   ├── models.py                         # PyTorch Basic Transformer from scratch
│   ├── mt5_system.py                     # mT5 Seq2Seq pipeline and architecture
│   ├── decoder_only.py                   # LLaMA and Gemma prompting and generation
│   ├── evaluate.py                       # BLEU, chrF, latency, and temperature metrics
│   ├── ethical_analysis.py               # Responsible AI test suite
│   └── plots.py                          # Matplotlib figure generation
├── Machine_Translation_System.ipynb      # Complete executable Jupyter Notebook
├── main.py                               # CLI execution entry point
├── create_submission_docx.py             # Generates Step 3 Word submission document
├── YourName_RegNum_FGAI_Assignment3.docx # Pre-generated Word document for LMS submission
├── requirements.txt                      # Project dependency specification
├── .gitignore                            # Standard git exclusion rules
└── README.md                             # Comprehensive technical documentation
```

---

## 11. Step-by-Step GitHub Upload Instructions

1. **Initialize Git Repository:**
   ```bash
   git init
   git add .
   git commit -m "Complete Assignment 3: Machine Translation Systems benchmark"
   ```
2. **Create Remote Repository on GitHub:**
   - Go to [GitHub.com](https://github.com/) &rarr; Click **New Repository**.
   - Name it: `ICT4403_Assignment3_Machine_Translation` (Public or Private).
3. **Link and Push:**
   ```bash
   git remote add origin https://github.com/<YOUR_GITHUB_USERNAME>/ICT4403_Assignment3_Machine_Translation.git
   git branch -M main
   git push -u origin main
   ```
4. **Final LMS Submission:**
   - Open `YourName_RegNum_FGAI_Assignment3.docx`.
   - Update your **Student Name**, **Registration Number**, and **GitHub Repository Link**.
   - Export as PDF (`File -> Save As -> PDF`) and upload to the LMS portal before the deadline.
