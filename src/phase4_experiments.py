# ============================================================================
# PHASE 4: PARAMETER EXPERIMENTATION
# Testing different embedding sizes and retrieval strategies
# ============================================================================

print("="*70)
print("PHASE 4: PARAMETER EXPERIMENTATION")
print("="*70)
print()

# ============================================================================
# IMPORTS
# ============================================================================
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss
from tqdm import tqdm
import json
from pathlib import Path
import time

print("✓ Imports successful\n")

# ============================================================================
# ASSUMPTION: You have from previous phases:
# - generator (LLM pipeline from Phase 3)
# - df_text (text corpus)
# - df_qa (QA dataset)
# - squad_metric (evaluation metric)
# - generate_answer() function using "basic" strategy
# ============================================================================

# ============================================================================
# CONFIGURATION: EXPERIMENTAL PARAMETERS
# ============================================================================

# Embedding models with different dimensions
EMBEDDING_MODELS = {
    "MiniLM-L6-v2-384": "all-MiniLM-L6-v2",  # 384 dimensions (baseline)
    "MiniLM-L12-v2-384": "all-MiniLM-L12-v2",  # 384 dimensions (deeper model)
    "MPNet-base-768": "all-mpnet-base-v2",  # 768 dimensions (larger)
}

# Retrieval strategies to test
RETRIEVAL_CONFIGS = {
    "top-1": {"k": 1, "strategy": "single"},
    "top-3": {"k": 3, "strategy": "concatenate"},
    "top-5": {"k": 5, "strategy": "concatenate"},
    "top-10": {"k": 10, "strategy": "concatenate"},
}

# Number of samples for evaluation
NUM_EVAL_SAMPLES = 100

print("Experimental Configuration:")
print(f"  Embedding models: {len(EMBEDDING_MODELS)}")
print(f"  Retrieval configs: {len(RETRIEVAL_CONFIGS)}")
print(f"  Total experiments: {len(EMBEDDING_MODELS) * len(RETRIEVAL_CONFIGS)}")
print(f"  Evaluation samples: {NUM_EVAL_SAMPLES}")
print()

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def build_index_for_model(model_name: str, model_path: str):
    """Build FAISS index for a specific embedding model."""
    print(f"\nBuilding index for {model_name}...")

    # Load embedding model
    embedding_model = SentenceTransformer(model_path)

    # Generate embeddings
    passage_texts = df_text['passage'].tolist()
    embeddings = embedding_model.encode(
        passage_texts,
        batch_size=32,
        show_progress_bar=True,
        convert_to_numpy=True,
        device='cpu'
    )

    # Normalize and build index
    faiss.normalize_L2(embeddings)
    embedding_dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(embedding_dim)
    index.add(embeddings)

    print(f"✓ Index built: {index.ntotal} vectors, {embedding_dim} dimensions")

    return embedding_model, index, embedding_dim


def retrieve_with_config(query: str, embedding_model, index, config: dict):
    """Retrieve passages using specified configuration."""
    k = config['k']
    strategy = config['strategy']

    # Encode query
    query_embedding = embedding_model.encode([query], convert_to_numpy=True)
    faiss.normalize_L2(query_embedding)

    # Search
    scores, indices = index.search(query_embedding, k)

    # Build context based on strategy
    if strategy == "single":
        # Just return top-1
        context = df_text.iloc[indices[0][0]]['passage']
    elif strategy == "concatenate":
        # Concatenate top-k passages
        passages = [df_text.iloc[idx]['passage'] for idx in indices[0]]
        context = "\n\n".join(passages)
    else:
        context = df_text.iloc[indices[0][0]]['passage']

    return context, float(scores[0][0])


def evaluate_configuration(model_name: str, embedding_model, index,
                          retrieval_name: str, retrieval_config: dict):
    """Evaluate a specific parameter configuration."""

    predictions = []
    references = []
    retrieval_scores = []

    for idx in tqdm(range(min(NUM_EVAL_SAMPLES, len(df_qa))),
                    desc=f"{model_name} + {retrieval_name}"):
        qa_pair = df_qa.iloc[idx]
        question = qa_pair['question']
        ground_truth = qa_pair['answer']

        try:
            # Retrieve with current config
            context, score = retrieve_with_config(
                question, embedding_model, index, retrieval_config
            )
            retrieval_scores.append(score)

            # Generate answer (using basic strategy from Phase 3)
            predicted_answer = generate_answer(question, context, "basic")

            predictions.append(predicted_answer)
            references.append(ground_truth)

        except Exception as e:
            print(f"\nError at idx {idx}: {e}")
            predictions.append("")
            references.append(ground_truth)
            retrieval_scores.append(0.0)

    # Calculate metrics
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
        "exact_match": results["exact_match"],
        "avg_retrieval_score": np.mean(retrieval_scores),
        "predictions": predictions,
        "references": references
    }


# ============================================================================
# RUN EXPERIMENTS
# ============================================================================

print("\n" + "="*70)
print("RUNNING EXPERIMENTS")
print("="*70)
print()

all_results = []
experiment_count = 0
total_experiments = len(EMBEDDING_MODELS) * len(RETRIEVAL_CONFIGS)

start_time = time.time()

for model_name, model_path in EMBEDDING_MODELS.items():
    print(f"\n{'='*70}")
    print(f"EMBEDDING MODEL: {model_name}")
    print(f"{'='*70}")

    # Build index for this embedding model
    embedding_model, index, embedding_dim = build_index_for_model(model_name, model_path)

    # Test all retrieval configurations
    for retrieval_name, retrieval_config in RETRIEVAL_CONFIGS.items():
        experiment_count += 1
        print(f"\n[Experiment {experiment_count}/{total_experiments}] "
              f"{model_name} + {retrieval_name}")

        # Run evaluation
        results = evaluate_configuration(
            model_name, embedding_model, index,
            retrieval_name, retrieval_config
        )

        # Store results
        all_results.append({
            "embedding_model": model_name,
            "embedding_dim": embedding_dim,
            "retrieval_strategy": retrieval_name,
            "top_k": retrieval_config['k'],
            "f1_score": results['f1'],
            "exact_match": results['exact_match'],
            "avg_retrieval_score": results['avg_retrieval_score']
        })

        print(f"  F1: {results['f1']:.2f} | EM: {results['exact_match']:.2f} | "
              f"Avg Retrieval: {results['avg_retrieval_score']:.4f}")

elapsed_time = time.time() - start_time
print(f"\n✓ All experiments completed in {elapsed_time/60:.1f} minutes")

# ============================================================================
# ANALYZE RESULTS
# ============================================================================

print("\n" + "="*70)
print("RESULTS ANALYSIS")
print("="*70)
print()

# Create results DataFrame
results_df = pd.DataFrame(all_results)

# Sort by F1 score
results_df_sorted = results_df.sort_values('f1_score', ascending=False)

print("Complete Results (sorted by F1 score):")
print(results_df_sorted.to_string(index=False))
print()

# Find best configuration
best_config = results_df_sorted.iloc[0]
print(" BEST CONFIGURATION:")
print(f"  Embedding: {best_config['embedding_model']}")
print(f"  Dimensions: {best_config['embedding_dim']}")
print(f"  Retrieval: {best_config['retrieval_strategy']}")
print(f"  F1 Score: {best_config['f1_score']:.2f}")
print(f"  Exact Match: {best_config['exact_match']:.2f}")
print()

# Analysis by embedding model
print("Performance by Embedding Model:")
embedding_comparison = results_df.groupby('embedding_model').agg({
    'f1_score': ['mean', 'max'],
    'exact_match': ['mean', 'max']
}).round(2)
print(embedding_comparison)
print()

# Analysis by retrieval strategy
print("Performance by Retrieval Strategy:")
retrieval_comparison = results_df.groupby('retrieval_strategy').agg({
    'f1_score': ['mean', 'max'],
    'exact_match': ['mean', 'max']
}).round(2)
print(retrieval_comparison)
print()

# ============================================================================
# SAVE RESULTS
# ============================================================================

print("="*70)
print("SAVING RESULTS")
print("="*70)
print()

Path("./results").mkdir(exist_ok=True)

# Save complete results
results_df.to_csv("./results/phase4_parameter_experiments.csv", index=False)
print("✓ Complete results: ./results/phase4_parameter_experiments.csv")

# Save analysis summaries
embedding_comparison.to_csv("./results/phase4_embedding_comparison.csv")
print("✓ Embedding comparison: ./results/phase4_embedding_comparison.csv")

retrieval_comparison.to_csv("./results/phase4_retrieval_comparison.csv")
print("✓ Retrieval comparison: ./results/phase4_retrieval_comparison.csv")

# Save best configuration
best_config_dict = {
    "best_configuration": {
        "embedding_model": best_config['embedding_model'],
        "embedding_dim": int(best_config['embedding_dim']),
        "retrieval_strategy": best_config['retrieval_strategy'],
        "top_k": int(best_config['top_k']),
        "f1_score": float(best_config['f1_score']),
        "exact_match": float(best_config['exact_match']),
        "avg_retrieval_score": float(best_config['avg_retrieval_score'])
    },
    "total_experiments": total_experiments,
    "evaluation_samples": NUM_EVAL_SAMPLES,
    "baseline_f1": 53.59,  # From Phase 3
    "improvement": float(best_config['f1_score'] - 53.59)
}

with open("./results/phase4_summary.json", "w") as f:
    json.dump(best_config_dict, f, indent=2)
print("✓ Summary: ./results/phase4_summary.json")
