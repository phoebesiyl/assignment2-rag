# ============================================================================
# PHASE 2: NAIVE RAG IMPLEMENTATION - COMPLETE PIPELINE
# ============================================================================

# Install dependencies
print("Installing dependencies...")
!pip install -q datasets sentence-transformers faiss-cpu PyYAML pandas
print("✓ Installation complete\n")

# ============================================================================
# IMPORTS
# ============================================================================
import numpy as np
import pandas as pd
from datasets import load_dataset
from sentence_transformers import SentenceTransformer
import faiss
import json
import yaml
import logging
from pathlib import Path
from typing import List, Dict, Optional
import sys


# ============================================================================
# LOGGING SETUP
# ============================================================================
def setup_logging(config: dict) -> logging.Logger:
    """Configure logging based on config file."""
    log_config = config['logging']

    if log_config['log_to_file']:
        Path(log_config['log_file']).parent.mkdir(parents=True, exist_ok=True)

    # Configure logging
    logging.basicConfig(
        level=getattr(logging, log_config['level']),
        format=log_config['format'],
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(log_config['log_file']) if log_config['log_to_file'] else logging.NullHandler()
        ]
    )

    return logging.getLogger(__name__)

# Load config and setup logging
try:
    with open("config.yaml", "r") as f:
        config = yaml.safe_load(f)
    logger = setup_logging(config)
    logger.info("Configuration loaded and logging initialized")
except Exception as e:
    print(f"ERROR: Failed to load configuration: {e}")
    sys.exit(1)

# ============================================================================
# NAIVE RAG CLASS WITH ERROR HANDLING
# ============================================================================
class NaiveRAG:
    """
    Naive RAG implementation with comprehensive error handling.
    """

    def __init__(self, config: dict, logger: logging.Logger):
        """Initialize RAG system."""
        self.config = config
        self.logger = logger
        self.embedding_model = None
        self.index = None
        self.df_text = None
        self.df_qa = None

        # Create required directories
        self._create_directories()

    def _create_directories(self):
        """Create necessary directories."""
        try:
            Path(self.config['dataset']['cache_dir']).mkdir(parents=True, exist_ok=True)
            Path(self.config['vector_db']['index_path']).parent.mkdir(parents=True, exist_ok=True)
            self.logger.info("Directories created successfully")
        except Exception as e:
            self.logger.error(f"Failed to create directories: {e}")
            raise

    def load_datasets(self) -> tuple:
        """
        Load text corpus and QA dataset with error handling.

        Returns:
            tuple: (df_text, df_qa)
        """
        try:
            self.logger.info("Loading text corpus...")
            ds_text = load_dataset(
                self.config['dataset']['name'],
                self.config['dataset']['corpus_split'],
                cache_dir=self.config['dataset']['cache_dir']
            )

            if not ds_text:
                raise ValueError("Text corpus is empty")

            text_split = next(iter(ds_text.keys()))
            self.df_text = pd.DataFrame(ds_text[text_split])

            # Validate corpus
            if self.df_text.empty:
                raise ValueError("Text corpus DataFrame is empty")
            if 'passage' not in self.df_text.columns or 'id' not in self.df_text.columns:
                raise ValueError("Text corpus missing required columns: 'passage' or 'id'")

            self.logger.info(f"Text corpus loaded: {self.df_text.shape[0]} passages")

        except Exception as e:
            self.logger.error(f"Failed to load text corpus: {e}")
            raise

        try:
            self.logger.info("Loading QA dataset...")
            ds_qa = load_dataset(
                self.config['dataset']['name'],
                self.config['dataset']['qa_split'],
                cache_dir=self.config['dataset']['cache_dir']
            )

            if not ds_qa:
                raise ValueError("QA dataset is empty")

            qa_split = next(iter(ds_qa.keys()))
            self.df_qa = pd.DataFrame(ds_qa[qa_split])

            # Validate QA dataset
            if self.df_qa.empty:
                raise ValueError("QA DataFrame is empty")
            required_cols = ['question', 'answer', 'id']
            if not all(col in self.df_qa.columns for col in required_cols):
                raise ValueError(f"QA dataset missing required columns: {required_cols}")

            self.logger.info(f"QA dataset loaded: {self.df_qa.shape[0]} pairs")

        except Exception as e:
            self.logger.error(f"Failed to load QA dataset: {e}")
            raise

        return self.df_text, self.df_qa

    def load_embedding_model(self):
        """Load sentence transformer model with error handling."""
        try:
            model_name = self.config['embedding']['model_name']
            self.logger.info(f"Loading embedding model: {model_name}")

            self.embedding_model = SentenceTransformer(model_name)

            # Verify embedding dimension
            test_embedding = self.embedding_model.encode(["test"])
            actual_dim = test_embedding.shape[1]
            expected_dim = self.config['embedding']['embedding_dim']

            if actual_dim != expected_dim:
                self.logger.warning(
                    f"Embedding dimension mismatch: expected {expected_dim}, got {actual_dim}"
                )
                self.config['embedding']['embedding_dim'] = actual_dim

            self.logger.info(f"Embedding model loaded (dim={actual_dim})")

        except Exception as e:
            self.logger.error(f"Failed to load embedding model: {e}")
            raise

    def create_embeddings(self) -> np.ndarray:
        """
        Generate embeddings with error handling and progress tracking.

        Returns:
            np.ndarray: Normalized embeddings
        """
        if self.df_text is None:
            raise ValueError("Text corpus not loaded. Call load_datasets() first.")

        if self.embedding_model is None:
            raise ValueError("Embedding model not loaded. Call load_embedding_model() first.")

        try:
            self.logger.info(f"Generating embeddings for {len(self.df_text)} passages...")

            passage_texts = self.df_text['passage'].tolist()

            if not passage_texts:
                raise ValueError("No passages to embed")

            embeddings = self.embedding_model.encode(
                passage_texts,
                batch_size=self.config['embedding']['batch_size'],
                show_progress_bar=True,
                convert_to_numpy=True,
                device=self.config['embedding']['device']
            )

            # Validate embeddings
            if embeddings is None or embeddings.size == 0:
                raise ValueError("Embedding generation returned empty array")

            self.logger.info(f"Embeddings generated: shape={embeddings.shape}")

            # Normalize if configured
            if self.config['embedding']['normalize']:
                self.logger.info("Normalizing embeddings...")
                faiss.normalize_L2(embeddings)

            return embeddings

        except Exception as e:
            self.logger.error(f"Failed to generate embeddings: {e}")
            raise

    def build_index(self, embeddings: np.ndarray):
        """
        Build FAISS index with error handling.

        Args:
            embeddings: Normalized embedding vectors
        """
        try:
            embedding_dim = self.config['embedding']['embedding_dim']
            index_type = self.config['vector_db']['index_type']

            self.logger.info(f"Building FAISS index: {index_type}")

            if index_type == "IndexFlatIP":
                self.index = faiss.IndexFlatIP(embedding_dim)
            elif index_type == "IndexFlatL2":
                self.index = faiss.IndexFlatL2(embedding_dim)
            else:
                self.logger.warning(f"Unknown index type '{index_type}', using IndexFlatIP")
                self.index = faiss.IndexFlatIP(embedding_dim)

            self.index.add(embeddings)

            # Verify index
            if self.index.ntotal != len(embeddings):
                raise ValueError(
                    f"Index size mismatch: expected {len(embeddings)}, got {self.index.ntotal}"
                )

            self.logger.info(f"FAISS index built: {self.index.ntotal} vectors")

            # Save index if configured
            if self.config['vector_db']['save_index']:
                self.save_index()

        except Exception as e:
            self.logger.error(f"Failed to build FAISS index: {e}")
            raise

    def save_index(self):
        """Save FAISS index and metadata with error handling."""
        try:
            index_path = self.config['vector_db']['index_path']
            metadata_path = self.config['vector_db']['metadata_path']

            self.logger.info(f"Saving index to {index_path}")

            # Ensure directory exists
            Path(index_path).parent.mkdir(parents=True, exist_ok=True)

            # Save FAISS index
            faiss.write_index(self.index, index_path)

            # Save metadata
            metadata = {
                'passage_ids': self.df_text['id'].tolist(),
                'embedding_dim': self.config['embedding']['embedding_dim'],
                'num_passages': len(self.df_text),
                'model': self.config['embedding']['model_name'],
                'index_type': self.config['vector_db']['index_type']
            }

            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2)

            self.logger.info("Index and metadata saved successfully")

        except Exception as e:
            self.logger.error(f"Failed to save index: {e}")
            raise

    def retrieve(self, query: str, top_k: Optional[int] = None) -> List[Dict]:
        """
        Retrieve top-k passages with error handling.

        Args:
            query: Search query
            top_k: Number of results (uses config default if None)

        Returns:
            List of retrieved passages with scores
        """
        if self.index is None:
            raise ValueError("Index not built. Call build_index() first.")

        if not query or not query.strip():
            raise ValueError("Query cannot be empty")

        try:
            if top_k is None:
                top_k = self.config['retrieval']['default_top_k']

            # Validate top_k
            if top_k <= 0:
                raise ValueError(f"top_k must be positive, got {top_k}")
            if top_k > self.index.ntotal:
                self.logger.warning(
                    f"top_k ({top_k}) > index size ({self.index.ntotal}), using {self.index.ntotal}"
                )
                top_k = self.index.ntotal

            # Encode query
            query_embedding = self.embedding_model.encode([query], convert_to_numpy=True)

            if self.config['embedding']['normalize']:
                faiss.normalize_L2(query_embedding)

            # Search
            scores, indices = self.index.search(query_embedding, top_k)

            # Build results
            threshold = self.config['retrieval']['similarity_threshold']
            results = []

            for idx, score in zip(indices[0], scores[0]):
                if score >= threshold:
                    result = {
                        'passage': self.df_text.iloc[idx]['passage'],
                        'id': int(self.df_text.iloc[idx]['id'])
                    }
                    if self.config['retrieval']['return_scores']:
                        result['score'] = float(score)
                    results.append(result)

            self.logger.debug(f"Retrieved {len(results)} passages for query: {query[:50]}...")
            return results

        except Exception as e:
            self.logger.error(f"Retrieval failed for query '{query[:50]}...': {e}")
            raise

# ============================================================================
# MAIN EXECUTION
# ============================================================================
print("="*70)
print("PHASE 2: NAIVE RAG IMPLEMENTATION")
print("="*70)
print()

try:
    # Initialize RAG
    logger.info("Initializing RAG system...")
    rag = NaiveRAG(config, logger)
    print("✓ RAG system initialized\n")

    # Load datasets
    logger.info("Loading datasets...")
    df_text, df_qa = rag.load_datasets()
    print(f"✓ Text corpus: {len(df_text)} passages")
    print(f"✓ QA dataset: {len(df_qa)} pairs\n")

    # Load embedding model
    rag.load_embedding_model()
    print(f"✓ Embedding model loaded\n")

    # Create embeddings
    embeddings = rag.create_embeddings()
    print(f"✓ Embeddings created: {embeddings.shape}\n")

    # Build index
    rag.build_index(embeddings)
    print(f"✓ FAISS index built: {rag.index.ntotal} vectors\n")

    # Test retrieval
    print("="*70)
    print("TESTING RETRIEVAL")
    print("="*70)
    print()

    test_queries = [
        "What is the capital of France?",
        "Who invented the telephone?",
        "When did World War II end?"
    ]

    for i, query in enumerate(test_queries, 1):
        print(f"Query {i}: {query}")
        results = rag.retrieve(query, top_k=3)

        for j, result in enumerate(results, 1):
            print(f"  [{j}] Score: {result['score']:.4f} | ID: {result['id']}")
            print(f"      {result['passage'][:120]}...")
        print()
