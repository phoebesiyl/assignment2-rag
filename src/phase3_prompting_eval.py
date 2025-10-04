# ============================================================================
# PHASE 3: EVALUATION PHASE I - Prompting Strategies & Metrics
# ============================================================================

# Install additional dependencies for LLM and evaluation
print("Installing dependencies for Phase 3...")
!pip install -q transformers torch evaluate datasets pandas numpy
print("✓ Installation complete\n")

# ============================================================================
# IMPORTS
# ============================================================================
import pandas as pd
import numpy as np
from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM
import evaluate
from typing import List, Dict
import json
from pathlib import Path
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

print("✓ Imports successful\n")


# ============================================================================
# STEP 1: LOAD LLM FOR ANSWER GENERATION
# ============================================================================
print("="*70)
print("STEP 1: LOADING LANGUAGE MODEL")
print("="*70)
print()

print("Loading Flan-T5-base for answer generation...")
print("(This may take 1-2 minutes...)")

try:
    # Load model and tokenizer
    model_name = "google/flan-t5-base"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

    # Create generation pipeline
    generator = pipeline(
        "text2text-generation",
        model=model,
        tokenizer=tokenizer,
        max_length=256,
        device=-1  # CPU; use 0 for GPU
    )

    print(f"✓ {model_name} loaded successfully\n")

except Exception as e:
    print(f"ERROR loading model: {e}")
    raise

# ============================================================================
# STEP 2: DEFINE PROMPTING STRATEGIES
# ============================================================================
print("="*70)
print("STEP 2: DEFINING PROMPTING STRATEGIES")
print("="*70)
print()

def create_prompt(question: str, context: str, strategy: str) -> str:
    """
    Create prompts using different strategies.

    Args:
        question: The question to answer
        context: Retrieved passage (top-1 document)
        strategy: Prompting strategy name

    Returns:
        Formatted prompt string
    """

    strategies = {
        "basic": f"""Context: {context}

Question: {question}

Answer:""",

        "cot": f"""Context: {context}

Question: {question}

Let's think step by step to answer this question based on the context:""",

        "persona": f"""You are a knowledgeable expert assistant. Use only the information provided in the context to answer accurately.

Context: {context}

Question: {question}

Expert Answer:""",

        "instruction": f"""Instructions: Read the context carefully and answer the question using ONLY information from the context. Be concise and accurate.

Context: {context}

Question: {question}

Answer:""",
    }

    return strategies.get(strategy, strategies["basic"])


def generate_answer(question: str, context: str, strategy: str) -> str:
    """
    Generate answer using LLM with specified prompting strategy.

    Args:
        question: Question to answer
        context: Retrieved passage
        strategy: Prompting strategy

    Returns:
        Generated answer string
    """
    prompt = create_prompt(question, context, strategy)

    try:
        result = generator(
            prompt,
            max_new_tokens=100,
            do_sample=False,  # Deterministic for evaluation
            temperature=1.0,
            num_beams=1
        )
        return result[0]['generated_text'].strip()

    except Exception as e:
        print(f"Generation error: {e}")
        return ""


print("✓ Prompting strategies defined:")
print("  1. Basic - Simple context-question-answer format")
print("  2. CoT - Chain-of-thought reasoning")
print("  3. Persona - Expert persona prompting")
print("  4. Instruction - Explicit instructions")
print()

# ============================================================================
# STEP 3: LOAD EVALUATION METRICS
# ============================================================================
print("="*70)
print("STEP 3: LOADING EVALUATION METRICS")
print("="*70)
print()

print("Loading SQuAD metric from HuggingFace...")
squad_metric = evaluate.load("squad")
print("✓ SQuAD metric loaded (provides F1 and Exact Match)\n")

def calculate_metrics(predictions: List[str], references: List[str]) -> Dict:
    """
    Calculate F1 and Exact Match scores using SQuAD metric.

    Args:
        predictions: List of predicted answers
        references: List of ground truth answers

    Returns:
        Dict with f1 and exact_match scores
    """
    # Format for SQuAD metric
    formatted_predictions = [
        {"id": str(i), "prediction_text": pred}
        for i, pred in enumerate(predictions)
    ]

    formatted_references = [
        {"id": str(i), "answers": {"text": [ref], "answer_start": [0]}}
        for i, ref in enumerate(references)
    ]

    results = squad_metric.compute(
        predictions=formatted_predictions,
        references=formatted_references
    )

    return {
        "f1": results["f1"],
        "exact_match": results["exact_match"]
    }

# ============================================================================
# STEP 4: RUN EVALUATION ON ALL STRATEGIES
# ============================================================================
print("="*70)
print("STEP 4: RUNNING EVALUATION")
print("="*70)
print()

# Configuration
NUM_EVAL_SAMPLES = 100  # Use subset for faster evaluation
strategies = ["basic", "cot", "persona", "instruction"]

print(f"Evaluating on {NUM_EVAL_SAMPLES} QA pairs")
print(f"Testing {len(strategies)} prompting strategies\n")

# Storage for results
all_results = []
strategy_scores = {}

# Evaluate each strategy
for strategy in strategies:
    print(f"\n{'='*70}")
    print(f"EVALUATING: {strategy.upper()} Strategy")
    print(f"{'='*70}\n")

    predictions = []
    references = []

    # Process each QA pair
    for idx in tqdm(range(min(NUM_EVAL_SAMPLES, len(df_qa))), desc=f"{strategy}"):
        qa_pair = df_qa.iloc[idx]
        question = qa_pair['question']
        ground_truth = qa_pair['answer']

        try:
            # Retrieve top-1 document
            retrieved = rag.retrieve(question, top_k=1)

            if not retrieved:
                predictions.append("")
                references.append(ground_truth)
                continue

            # Get top-1 context
            context = retrieved[0]['passage']

            # Generate answer with strategy
            predicted_answer = generate_answer(question, context, strategy)

            predictions.append(predicted_answer)
            references.append(ground_truth)

            # Store detailed results
            all_results.append({
                'strategy': strategy,
                'question': question,
                'ground_truth': ground_truth,
                'predicted_answer': predicted_answer,
                'retrieval_score': retrieved[0]['score'],
                'context': context[:200] + "..."
            })

        except Exception as e:
            print(f"\nError processing question {idx}: {e}")
            predictions.append("")
            references.append(ground_truth)

    # Calculate metrics for this strategy
    metrics = calculate_metrics(predictions, references)
    strategy_scores[strategy] = metrics

    print(f"\n{strategy.upper()} Results:")
    print(f"  F1 Score: {metrics['f1']:.2f}")
    print(f"  Exact Match: {metrics['exact_match']:.2f}")

# ============================================================================
# STEP 5: ANALYZE AND COMPARE RESULTS
# ============================================================================
print("\n" + "="*70)
print("STEP 5: RESULTS COMPARISON")
print("="*70)
print()

# Create comparison DataFrame
comparison_df = pd.DataFrame([
    {
        'Strategy': strategy,
        'F1 Score': scores['f1'],
        'Exact Match': scores['exact_match']
    }
    for strategy, scores in strategy_scores.items()
])

# Sort by F1 score
comparison_df = comparison_df.sort_values('F1 Score', ascending=False)

print("Performance Comparison:")
print(comparison_df.to_string(index=False))
print()

# Identify best strategy
best_strategy = comparison_df.iloc[0]['Strategy']
best_f1 = comparison_df.iloc[0]['F1 Score']
best_em = comparison_df.iloc[0]['Exact Match']

print(f" BEST PERFORMING STRATEGY: {best_strategy.upper()}")
print(f"   F1 Score: {best_f1:.2f}")
print(f"   Exact Match: {best_em:.2f}")
print()

# ============================================================================
# STEP 6: SAVE RESULTS
# ============================================================================
print("="*70)
print("STEP 6: SAVING RESULTS")
print("="*70)
print()

# Create results directory
Path("./results").mkdir(exist_ok=True)

# Save detailed results
results_df = pd.DataFrame(all_results)
results_df.to_csv("./results/phase3_detailed_results.csv", index=False)
print("✓ Detailed results saved: ./results/phase3_detailed_results.csv")

# Save comparison
comparison_df.to_csv("./results/phase3_strategy_comparison.csv", index=False)
print("✓ Strategy comparison saved: ./results/phase3_strategy_comparison.csv")

# Save summary JSON
summary = {
    "evaluation_samples": NUM_EVAL_SAMPLES,
    "strategies_tested": strategies,
    "best_strategy": best_strategy,
    "strategy_scores": strategy_scores,
    "best_performance": {
        "f1": float(best_f1),
        "exact_match": float(best_em)
    }
}

with open("./results/phase3_summary.json", "w") as f:
    json.dump(summary, f, indent=2)
print("✓ Summary saved: ./results/phase3_summary.json")

# ============================================================================
# STEP 7: SAMPLE RESULTS ANALYSIS
# ============================================================================
print("\n" + "="*70)
print("STEP 7: SAMPLE RESULTS")
print("="*70)
print()

print(f"Showing 3 examples from BEST strategy ({best_strategy}):\n")

best_strategy_results = results_df[results_df['strategy'] == best_strategy].head(3)

for idx, row in best_strategy_results.iterrows():
    print(f"Example {idx + 1}:")
    print(f"  Question: {row['question']}")
    print(f"  Ground Truth: {row['ground_truth']}")
    print(f"  Predicted: {row['predicted_answer']}")
    print(f"  Retrieval Score: {row['retrieval_score']:.4f}")
    print()
