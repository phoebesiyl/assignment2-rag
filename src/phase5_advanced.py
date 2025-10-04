# ============================================================================
# PHASE 5: ADVANCED RAG FEATURES
# Implementing Query Rewriting and Reranking
# ============================================================================

print("="*70)
print("PHASE 5: ADVANCED RAG FEATURES")
print("="*70)
print()

# ============================================================================
# INSTALL ADDITIONAL DEPENDENCIES
# ============================================================================
print("Installing dependencies for advanced features...")
!pip install -q rank-bm25 nltk
print("✓ Installation complete\n")

# ============================================================================
# IMPORTS
# ============================================================================
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer, CrossEncoder
import faiss
from rank_bm25 import BM25Okapi
import nltk
from nltk.tokenize import word_tokenize
from tqdm import tqdm
import json
from pathlib import Path
import re

# Download NLTK data
nltk.download('punkt', quiet=True)

print("✓ Imports successful\n")

# ============================================================================
# From Phase 4:
# - embedding_model (MiniLM-L6-v2)
# - index (FAISS index)
# - df_text (text corpus)
# - df_qa (QA dataset)
# - generator (LLM pipeline)
# - squad_metric
# ============================================================================

# ============================================================================
# FEATURE 1: QUERY REWRITING
# ============================================================================

print("="*70)
print("FEATURE 1: QUERY REWRITING")
print("="*70)
print()

class QueryRewriter:
    """
    Rewrites queries to improve retrieval through multiple strategies.
    """
    
    def __init__(self, generator_pipeline):
        self.generator = generator_pipeline
        print("✓ Query Rewriter initialized")
    
    def rewrite_with_llm(self, query: str) -> list:
        """Generate query variations using LLM."""
        prompt = f"""Generate 2 alternative phrasings of this question that mean the same thing but use different words:

Original question: {query}

Alternative 1:"""
        
        try:
            result = self.generator(
                prompt,
                max_new_tokens=100,
                do_sample=True,
                temperature=0.7,
                num_return_sequences=1
            )
            
            rewritten = result[0]['generated_text'].strip()
            
            # Extract alternatives (simple split approach)
            alternatives = [query]  # Always include original
            if rewritten:
                # Take first sentence as alternative
                alt = rewritten.split('\n')[0].strip()
                if alt and len(alt) > 5:
                    alternatives.append(alt)
            
            return alternatives
            
        except Exception as e:
            print(f"Rewriting error: {e}")
            return [query]
    
    def expand_query(self, query: str) -> str:
        """Simple query expansion by adding context."""
        # Add common Wikipedia context terms
        if '?' in query:
            expanded = query.replace('?', ' in history?')
        else:
            expanded = query + " information"
        return expanded


print("Testing Query Rewriting:")
rewriter = QueryRewriter(generator)

test_query = "What is the capital of France?"
rewrites = rewriter.rewrite_with_llm(test_query)
print(f"\nOriginal: {test_query}")
print(f"Rewrites: {rewrites}")
print()

# ============================================================================
# FEATURE 2: RERANKING
# ============================================================================

print("="*70)
print("FEATURE 2: RERANKING")
print("="*70)
print()

class Reranker:
    """
    Rerank retrieved passages using cross-encoder for better relevance.
    """
    
    def __init__(self, model_name='cross-encoder/ms-marco-MiniLM-L-6-v2'):
        print(f"Loading cross-encoder model: {model_name}")
        self.cross_encoder = CrossEncoder(model_name)
        print("✓ Reranker initialized")
    
    def rerank(self, query: str, passages: list, top_k: int = 5) -> list:
        """
        Rerank passages using cross-encoder scores.
        
        Args:
            query: Search query
            passages: List of dicts with 'passage' and 'score' keys
            top_k: Number of passages to return after reranking
        
        Returns:
            Reranked list of passages
        """
        if not passages:
            return passages
        
        # Prepare pairs for cross-encoder
        pairs = [(query, p['passage']) for p in passages]
        
        # Get cross-encoder scores
        ce_scores = self.cross_encoder.predict(pairs)
        
        # Add cross-encoder scores to passages
        for i, passage in enumerate(passages):
            passage['ce_score'] = float(ce_scores[i])
        
        # Sort by cross-encoder score and return top_k
        reranked = sorted(passages, key=lambda x: x['ce_score'], reverse=True)
        return reranked[:top_k]


print("Loading reranker model...")
reranker = Reranker()
print()

# ============================================================================
# ENHANCED RAG SYSTEM
# ============================================================================

print("="*70)
print("BUILDING ENHANCED RAG SYSTEM")
print("="*70)
print()

class EnhancedRAG:
    """
    RAG system with query rewriting and reranking.
    """
    
    def __init__(self, embedding_model, index, rewriter, reranker):
        self.embedding_model = embedding_model
        self.index = index
        self.rewriter = rewriter
        self.reranker = reranker
        print("✓ Enhanced RAG system initialized")
    
    def retrieve_and_rerank(self, query: str, initial_k: int = 10, 
                           final_k: int = 5, use_rewriting: bool = True) -> list:
        """
        Retrieve with optional query rewriting and mandatory reranking.
        
        Args:
            query: Original query
            initial_k: Number of passages to retrieve initially
            final_k: Number of passages after reranking
            use_rewriting: Whether to use query rewriting
        
        Returns:
            List of reranked passages
        """
        queries = [query]
        
        # Query rewriting
        if use_rewriting:
            queries = self.rewriter.rewrite_with_llm(query)
        
        # Retrieve for all query variations
        all_passages = {}
        
        for q in queries:
            # Encode query
            query_embedding = self.embedding_model.encode([q], convert_to_numpy=True)
            faiss.normalize_L2(query_embedding)
            
            # Search
            scores, indices = self.index.search(query_embedding, initial_k)
            
            # Collect unique passages
            for idx, score in zip(indices[0], scores[0]):
                passage_id = int(df_text.iloc[idx]['id'])
                if passage_id not in all_passages:
                    all_passages[passage_id] = {
                        'passage': df_text.iloc[idx]['passage'],
                        'id': passage_id,
                        'score': float(score)
                    }
        
        # Convert to list
        passages_list = list(all_passages.values())
        
        # Rerank
        reranked = self.reranker.rerank(query, passages_list, top_k=final_k)
        
        return reranked
    
    def generate_answer(self, query: str, passages: list) -> str:
        """Generate answer from reranked passages."""
        # Concatenate top passages
        context = "\n\n".join([p['passage'] for p in passages])
        
        # Use basic prompt (best from Phase 3)
        prompt = f"""Context: {context}

Question: {query}

Answer:"""
        
        try:
            result = generator(
                prompt,
                max_new_tokens=100,
                do_sample=False,
                temperature=1.0
            )
            return result[0]['generated_text'].strip()
        except Exception as e:
            print(f"Generation error: {e}")
            return ""


enhanced_rag = EnhancedRAG(embedding_model, index, rewriter, reranker)
print()

# ============================================================================
# EVALUATION: COMPARE SYSTEMS
# ============================================================================

print("="*70)
print("EVALUATION: COMPARING RAG SYSTEMS")
print("="*70)
print()

NUM_EVAL_SAMPLES = 100

def evaluate_system(system_name: str, use_rewriting: bool, use_reranking: bool):
    """Evaluate a RAG configuration."""
    predictions = []
    references = []
    
    print(f"\nEvaluating: {system_name}")
    
    for idx in tqdm(range(min(NUM_EVAL_SAMPLES, len(df_qa))), desc=system_name):
        qa_pair = df_qa.iloc[idx]
        question = qa_pair['question']
        ground_truth = qa_pair['answer']
        
        try:
            if use_reranking:
                # Use enhanced system
                passages = enhanced_rag.retrieve_and_rerank(
                    question, 
                    initial_k=10, 
                    final_k=5,
                    use_rewriting=use_rewriting
                )
                answer = enhanced_rag.generate_answer(question, passages)
            else:
                # Baseline: top-10 from Phase 4
                query_embedding = embedding_model.encode([question], convert_to_numpy=True)
                faiss.normalize_L2(query_embedding)
                scores, indices = index.search(query_embedding, 10)
                
                passages = [df_text.iloc[idx]['passage'] for idx in indices[0]]
                context = "\n\n".join(passages)
                
                prompt = f"Context: {context}\n\nQuestion: {question}\n\nAnswer:"
                result = generator(prompt, max_new_tokens=100, do_sample=False)
                answer = result[0]['generated_text'].strip()
            
            predictions.append(answer)
            references.append(ground_truth)
            
        except Exception as e:
            print(f"\nError at {idx}: {e}")
            predictions.append("")
            references.append(ground_truth)
    
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
        "system": system_name,
        "f1": results["f1"],
        "exact_match": results["exact_match"]
    }


# Run evaluations
systems_to_test = [
    ("Baseline (top-10)", False, False),
    ("With Reranking Only", False, True),
    ("With Query Rewriting + Reranking", True, True),
]

all_results = []

for system_name, use_rewriting, use_reranking in systems_to_test:
    result = evaluate_system(system_name, use_rewriting, use_reranking)
    all_results.append(result)
    print(f"\n{system_name}:")
    print(f"  F1: {result['f1']:.2f}")
    print(f"  EM: {result['exact_match']:.2f}")

# ============================================================================
# RESULTS ANALYSIS
# ============================================================================

print("\n" + "="*70)
print("RESULTS COMPARISON")
print("="*70)
print()

results_df = pd.DataFrame(all_results)
print("System Performance:")
print(results_df.to_string(index=False))
print()

# Calculate improvements
baseline_f1 = results_df.iloc[0]['f1']
best_f1 = results_df['f1'].max()
improvement = best_f1 - baseline_f1

print(f"Baseline F1: {baseline_f1:.2f}")
print(f"Best Enhanced F1: {best_f1:.2f}")
print(f"Improvement: {improvement:.2f} points")
print()

# ============================================================================
# SAVE RESULTS
# ============================================================================

print("="*70)
print("SAVING RESULTS")
print("="*70)
print()

Path("./results").mkdir(exist_ok=True)

# Save comparison
results_df.to_csv("./results/phase5_system_comparison.csv", index=False)
print("✓ System comparison: ./results/phase5_system_comparison.csv")

# Save summary
summary = {
    "baseline": {
        "name": "Phase 4 best (MiniLM-L6-v2 + top-10)",
        "f1": float(baseline_f1),
        "exact_match": float(results_df.iloc[0]['exact_match'])
    },
    "enhanced_systems": [
        {
            "name": row['system'],
            "f1": float(row['f1']),
            "exact_match": float(row['exact_match']),
            "improvement": float(row['f1'] - baseline_f1)
        }
        for _, row in results_df.iterrows()
    ],
    "best_system": results_df.loc[results_df['f1'].idxmax()]['system'],
    "total_improvement": float(improvement),
    "features_implemented": ["Query Rewriting", "Reranking"]
}

with open("./results/phase5_summary.json", "w") as f:
    json.dump(summary, f, indent=2)
print("✓ Summary: ./results/phase5_summary.json")
