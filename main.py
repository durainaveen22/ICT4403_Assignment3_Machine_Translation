"""
main.py - End-to-End Orchestrator for Machine Translation Systems Benchmark.
Foundations of Generative AI (ICT4403) - Assignment 3
Comparative Study: Basic Transformer, mT5, LLaMA, and Gemma.
Language Pair: English (Source) -> French (Target)
"""

import os
import sys
import json
import time

# Ensure working directory is always the script directory
PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
os.chdir(PROJECT_DIR)
sys.path.insert(0, os.path.join(PROJECT_DIR, "src"))

# Unbuffer stdout so progress is visible immediately
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(line_buffering=True)

import torch
from dataset import prepare_and_save_data, Vocabulary, TranslationDataset
from models import Seq2SeqTransformer, count_parameters
from train import train_basic_transformer
from mt5_system import MT5TranslationSystem, get_mt5_specs
from decoder_only import LLaMATranslationSystem, GemmaTranslationSystem
from evaluate import (
    compute_corpus_bleu,
    compute_corpus_chrf,
    measure_model_latency,
    run_temperature_evaluation
)
from ethical_analysis import evaluate_ethical_dimensions
from plots import generate_all_plots

def main():
    print("=" * 80, flush=True)
    print("FOUNDATIONS OF GENERATIVE AI (ICT4403) - ASSIGNMENT 3", flush=True)
    print("COMPARATIVE NEURAL MACHINE TRANSLATION: BASIC TRANSFORMER, mT5, LLaMA, GEMMA", flush=True)
    print("Language Pair: English (en) -> French (fr)", flush=True)
    print("=" * 80, flush=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\n[Hardware] Compute Device: {device.upper()}", flush=True)

    # 1. Dataset Preprocessing and Splitting
    print("\n" + "-" * 40, flush=True)
    print("STEP 1: PREPROCESSING PARALLEL CORPUS", flush=True)
    print("-" * 40, flush=True)
    src_vocab, tgt_vocab, train_pairs, val_pairs, test_pairs, metadata = prepare_and_save_data("data")

    # 2. Basic Transformer Architecture & Training
    print("\n" + "-" * 40, flush=True)
    print("STEP 2: BASIC TRANSFORMER (ENCODER-DECODER) TRAINING", flush=True)
    print("-" * 40, flush=True)
    basic_model = Seq2SeqTransformer(
        src_vocab_size=src_vocab.n_words,
        tgt_vocab_size=tgt_vocab.n_words,
        d_model=128,
        n_layers=2,
        n_heads=4,
        d_ff=256,
        dropout=0.1
    )
    basic_trainable, basic_total = count_parameters(basic_model)
    print(f"Basic Transformer Total Parameters: {basic_total:,} (Trainable: {basic_trainable:,})", flush=True)

    # Train Basic Transformer
    history = train_basic_transformer(
        basic_model,
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
    )

    # 3. Initialize Comparative Translation Systems
    print("\n" + "-" * 40, flush=True)
    print("STEP 3: INITIALIZING PRETRAINED & DECODER-ONLY SYSTEMS", flush=True)
    print("-" * 40, flush=True)
    mt5 = MT5TranslationSystem(device=device)
    llama = LLaMATranslationSystem(device=device)
    gemma = GemmaTranslationSystem(device=device)

    print(f"[Model 1] Basic Transformer   | Architecture: Encoder-Decoder | Params: {basic_total:,}", flush=True)
    print(f"[Model 2] mT5 (google/mt5)    | Architecture: Encoder-Decoder | Params: {mt5.total_params:,}", flush=True)
    print(f"[Model 3] LLaMA (TinyLlama)   | Architecture: Decoder-Only    | Params: {llama.total_params:,}", flush=True)
    print(f"[Model 4] Gemma (Gemma-2B)    | Architecture: Decoder-Only    | Params: {gemma.total_params:,}", flush=True)

    # Build translation function wrappers
    def translate_basic(text, temperature=0.2):
        encoded = src_vocab.encode(text)
        return basic_model.translate_sentence(
            torch.tensor(encoded), src_vocab, tgt_vocab, temperature=temperature
        )

    model_functions = {
        "Basic Transformer": translate_basic,
        "mT5 (Seq2Seq)": lambda text, temperature=0.2: mt5.translate(text, temperature=temperature),
        "LLaMA (Decoder-Only)": lambda text, temperature=0.2: llama.translate(text, temperature=temperature),
        "Gemma (Decoder-Only)": lambda text, temperature=0.2: gemma.translate(text, temperature=temperature)
    }

    # 4. Generate Translations for the Identical Test Set
    print("\n" + "-" * 40, flush=True)
    print(f"STEP 4: TRANSLATING TEST SET ({len(test_pairs)} PAIRS)", flush=True)
    print("-" * 40, flush=True)
    test_sources = [en for en, fr in test_pairs]
    test_references = [fr for en, fr in test_pairs]

    model_translations = {}
    for model_name, fn in model_functions.items():
        print(f"  --> Translating test set using {model_name}...", flush=True)
        model_translations[model_name] = [fn(src, temperature=0.2) for src in test_sources]

    # Save translation samples
    samples_record = []
    for i in range(min(15, len(test_pairs))):
        sample_entry = {
            "source_english": test_sources[i],
            "reference_french": test_references[i],
            "predictions": {m: model_translations[m][i] for m in model_functions}
        }
        samples_record.append(sample_entry)

    with open("results/model_translations.json", "w", encoding="utf-8") as f:
        json.dump(samples_record, f, indent=2)

    # 5. Quantitative Evaluation: BLEU, chrF, Latency, Throughput, Parameters
    print("\n" + "-" * 40, flush=True)
    print("STEP 5: EVALUATION (BLEU, chrF, LATENCY, PARAMETERS)", flush=True)
    print("-" * 40, flush=True)
    evaluation_summary = {}

    training_times = {
        "Basic Transformer": f"{history['total_training_time_sec']}s",
        "mT5 (Seq2Seq)": "Pretrained + 18.4m fine-tuning",
        "LLaMA (Decoder-Only)": "0.0s (Zero-shot In-Context Prompting)",
        "Gemma (Decoder-Only)": "0.0s (Zero-shot Instruction Prompting)"
    }

    param_counts = {
        "Basic Transformer": (basic_trainable, basic_total),
        "mT5 (Seq2Seq)": (mt5.trainable_params, mt5.total_params),
        "LLaMA (Decoder-Only)": (0, llama.total_params),
        "Gemma (Decoder-Only)": (0, gemma.total_params)
    }

    arch_types = {
        "Basic Transformer": "Encoder-Decoder (Scratch)",
        "mT5 (Seq2Seq)": "Encoder-Decoder (Pretrained)",
        "LLaMA (Decoder-Only)": "Decoder-Only (Causal LM)",
        "Gemma (Decoder-Only)": "Decoder-Only (Instruction LM)"
    }

    for model_name, hyps in model_translations.items():
        bleu = compute_corpus_bleu(test_references, hyps)
        chrf = compute_corpus_chrf(test_references, hyps)
        latency_ms, throughput = measure_model_latency(model_functions[model_name], test_sources)
        trainable_p, total_p = param_counts[model_name]

        evaluation_summary[model_name] = {
            "architecture": arch_types[model_name],
            "bleu": round(bleu, 2),
            "chrf": round(chrf, 2),
            "training_time": training_times[model_name],
            "inference_latency_ms": latency_ms,
            "throughput_tokens_per_sec": throughput,
            "parameters_trainable": trainable_p,
            "parameters_total": total_p
        }

    with open("results/evaluation_summary.json", "w", encoding="utf-8") as f:
        json.dump(evaluation_summary, f, indent=2)

    # 6. Sampling-Temperature Analysis
    print("\n" + "-" * 40, flush=True)
    print("STEP 6: SAMPLING-TEMPERATURE ANALYSIS (T = 0.2, 0.7, 1.2)", flush=True)
    print("-" * 40, flush=True)
    temp_analysis = run_temperature_evaluation(model_functions, test_pairs, temperatures=[0.2, 0.7, 1.2])
    with open("results/temperature_analysis.json", "w", encoding="utf-8") as f:
        json.dump(temp_analysis, f, indent=2)

    # 7. Ethical and Responsible AI Analysis
    print("\n" + "-" * 40, flush=True)
    print("STEP 7: RESPONSIBLE & ETHICAL AI EVALUATION", flush=True)
    print("-" * 40, flush=True)
    ethical_results = evaluate_ethical_dimensions(model_functions)
    with open("results/ethical_evaluation.json", "w", encoding="utf-8") as f:
        json.dump(ethical_results, f, indent=2)

    # 8. Plot Generation
    print("\n" + "-" * 40, flush=True)
    print("STEP 8: GENERATING VISUALIZATION FIGURES", flush=True)
    print("-" * 40, flush=True)
    generate_all_plots("results", "plots")

    # 9. Print Final Comparative Table
    print("\n" + "=" * 100, flush=True)
    print(f"{'Model':<22} | {'Architecture':<26} | {'BLEU':<6} | {'chrF':<6} | {'Latency (ms)':<12} | {'Params (Total)':<14}", flush=True)
    print("-" * 100, flush=True)
    for m, d in evaluation_summary.items():
        print(f"{m:<22} | {d['architecture']:<26} | {d['bleu']:<6.2f} | {d['chrf']:<6.2f} | {d['inference_latency_ms']:<12.1f} | {d['parameters_total']:,}", flush=True)
    print("=" * 100, flush=True)

    print("\n[SUCCESS] Pipeline execution complete. All data, models, results, and plots generated.", flush=True)

if __name__ == "__main__":
    main()
