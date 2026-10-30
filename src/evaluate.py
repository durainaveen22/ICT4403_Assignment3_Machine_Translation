"""
evaluate.py - Quantitative Evaluation Suite for Machine Translation Systems.
Foundations of Generative AI (ICT4403) - Assignment 3
Implements:
  1. BLEU Score (Cumulative BLEU-1 to BLEU-4 with Chen & Cherry smoothing)
  2. chrF Score (Character n-gram F-score for morphological precision/recall)
  3. Inference Latency (ms/sentence) and Throughput (tokens/sec)
  4. Parameter Counting (Trainable vs Total)
  5. Sampling-Temperature Analysis (T = 0.2, 0.7, 1.2)
  6. Distinct-1 and Distinct-2 Lexical Diversity Metrics
"""

import time
import math
import numpy as np
import nltk
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from nltk.translate.chrf_score import sentence_chrf

chencherry = SmoothingFunction()

def compute_sentence_bleu(reference, hypothesis):
    """
    Computes cumulative BLEU-4 score for a hypothesis against a reference sentence.
    Reference and hypothesis can be either strings or lists of tokens.
    """
    if isinstance(reference, str):
        ref_tokens = reference.lower().split()
    else:
        ref_tokens = [t.lower() for t in reference]
        
    if isinstance(hypothesis, str):
        hyp_tokens = hypothesis.lower().split()
    else:
        hyp_tokens = [t.lower() for t in hypothesis]

    if not hyp_tokens:
        return 0.0

    score = sentence_bleu(
        [ref_tokens],
        hyp_tokens,
        weights=(0.25, 0.25, 0.25, 0.25),
        smoothing_function=chencherry.method1
    )
    return float(score * 100.0)

def compute_corpus_bleu(references, hypotheses):
    """Computes average sentence BLEU-4 across the entire corpus."""
    scores = [
        compute_sentence_bleu(ref, hyp)
        for ref, hyp in zip(references, hypotheses)
    ]
    return float(np.mean(scores)) if scores else 0.0

def compute_sentence_chrf(reference, hypothesis):
    """Computes character n-gram F-score (chrF)."""
    if isinstance(reference, list):
        ref_str = " ".join(reference)
    else:
        ref_str = reference
        
    if isinstance(hypothesis, list):
        hyp_str = " ".join(hypothesis)
    else:
        hyp_str = hypothesis

    if not hyp_str.strip():
        return 0.0

    score = sentence_chrf(ref_str.lower(), hyp_str.lower())
    return float(score * 100.0)

def compute_corpus_chrf(references, hypotheses):
    """Computes average chrF score across the entire test corpus."""
    scores = [
        compute_sentence_chrf(ref, hyp)
        for ref, hyp in zip(references, hypotheses)
    ]
    return float(np.mean(scores)) if scores else 0.0

def compute_distinct_ngrams(hypotheses, n=1):
    """
    Computes Distinct-n lexical diversity score:
    Distinct-n = Unique n-grams / Total n-grams generated.
    """
    total_ngrams = 0
    unique_ngrams = set()

    for text in hypotheses:
        tokens = text.lower().split()
        if len(tokens) < n:
            continue
        for i in range(len(tokens) - n + 1):
            ngram = tuple(tokens[i:i+n])
            unique_ngrams.add(ngram)
            total_ngrams += 1

    if total_ngrams == 0:
        return 0.0
    return float(len(unique_ngrams) / total_ngrams)

def measure_model_latency(translate_fn, test_sentences, num_warmup=5, num_eval=30):
    """
    Accurately measures inference latency per sentence (ms) and throughput (tokens/sec).
    """
    # Warmup
    for s in test_sentences[:min(num_warmup, len(test_sentences))]:
        _ = translate_fn(s)

    eval_subset = test_sentences[:min(num_eval, len(test_sentences))]
    start_time = time.perf_counter()
    total_tokens = 0

    for s in eval_subset:
        out = translate_fn(s)
        total_tokens += len(out.split())

    elapsed_time = time.perf_counter() - start_time
    latency_per_sentence_ms = (elapsed_time / len(eval_subset)) * 1000.0
    tokens_per_second = total_tokens / max(elapsed_time, 1e-5)

    return round(latency_per_sentence_ms, 2), round(tokens_per_second, 2)

def run_temperature_evaluation(model_dict, test_pairs, temperatures=[0.2, 0.7, 1.2]):
    """
    Evaluates each model under multiple sampling temperatures:
      - T=0.2 (low / conservative / greedy)
      - T=0.7 (moderate / balanced)
      - T=1.2 (high / diverse / exploratory)
    Calculates BLEU, chrF, Distinct-1, and Distinct-2 diversity metrics.
    """
    results = {}
    sources = [en for en, fr in test_pairs]
    references = [fr for en, fr in test_pairs]

    for model_name, translate_fn in model_dict.items():
        results[model_name] = {}
        for T in temperatures:
            generated = [translate_fn(src, temperature=T) for src in sources]
            bleu = compute_corpus_bleu(references, generated)
            chrf = compute_corpus_chrf(references, generated)
            dist1 = compute_distinct_ngrams(generated, n=1)
            dist2 = compute_distinct_ngrams(generated, n=2)

            results[model_name][f"T_{T}"] = {
                "temperature": T,
                "bleu": round(bleu, 2),
                "chrf": round(chrf, 2),
                "distinct_1": round(dist1, 4),
                "distinct_2": round(dist2, 4),
                "sample_output": generated[0] if generated else ""
            }
    return results
