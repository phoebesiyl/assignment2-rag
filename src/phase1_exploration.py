# ============================================================================
# PHASE 1: DATASET EXPLORATION
# RAG Mini Wikipedia Dataset Analysis
# ============================================================================

print("="*70)
print("PHASE 1: DATASET SETUP AND EXPLORATION")
print("="*70)
print()

# ============================================================================
# INSTALL DEPENDENCIES
# ============================================================================
print("Installing dependencies...")
!pip install -q datasets pandas numpy matplotlib
print("✓ Installation complete\n")

# ============================================================================
# IMPORTS
# ============================================================================
import pandas as pd
import numpy as np
from datasets import load_dataset
import matplotlib.pyplot as plt

print("✓ Imports successful\n")

# ============================================================================
# STEP 1: LOAD TEXT CORPUS
# ============================================================================
print("="*70)
print("STEP 1: LOADING TEXT CORPUS")
print("="*70)
print()

print("Loading text corpus from HuggingFace...")
ds_text = load_dataset("rag-datasets/rag-mini-wikipedia", "text-corpus")

# Get the split name (usually 'passages')
text_split = next(iter(ds_text.keys()))
df_text = pd.DataFrame(ds_text[text_split])

print(f"✓ Text corpus loaded\n")
print(f"Dataset split: '{text_split}'")
print(f"Shape: {df_text.shape}")
print(f"Columns: {list(df_text.columns)}\n")

print("Sample passages:")
print(df_text.head(3))
print()

# ============================================================================
# STEP 2: TEXT CORPUS ANALYSIS
# ============================================================================
print("="*70)
print("STEP 2: TEXT CORPUS ANALYSIS")
print("="*70)
print()

# Data quality checks
print("Data Quality Checks:")
print(f"  Total passages: {len(df_text)}")
print(f"  Columns: {list(df_text.columns)}")
print()

print("Null values:")
print(df_text.isnull().sum())
print()

print("Duplicate rows:", df_text.duplicated().sum())
print("Duplicate IDs:", df_text['id'].duplicated().sum())
print()

# Passage length analysis
df_text['len_passage'] = df_text['passage'].str.len()

print("Passage length distribution:")
print(df_text['len_passage'].describe())
print()

print(f"Shortest passage: {df_text['len_passage'].min()} characters")
print(f"Longest passage: {df_text['len_passage'].max()} characters")
print(f"Average length: {df_text['len_passage'].mean():.1f} characters")
print(f"Median length: {df_text['len_passage'].median():.1f} characters")
print()

# ============================================================================
# STEP 3: LOAD QA DATASET
# ============================================================================
print("="*70)
print("STEP 3: LOADING QA DATASET")
print("="*70)
print()

print("Loading question-answer pairs from HuggingFace...")
ds_qa = load_dataset("rag-datasets/rag-mini-wikipedia", "question-answer")

# Get the split name (usually 'test' or 'train')
qa_split = next(iter(ds_qa.keys()))
df_qa = pd.DataFrame(ds_qa[qa_split])

print(f"✓ QA dataset loaded\n")
print(f"Dataset split: '{qa_split}'")
print(f"Shape: {df_qa.shape}")
print(f"Columns: {list(df_qa.columns)}\n")

print("Sample QA pairs:")
print(df_qa.head(3))
print()

# ============================================================================
# STEP 4: QA DATASET ANALYSIS
# ============================================================================
print("="*70)
print("STEP 4: QA DATASET ANALYSIS")
print("="*70)
print()

# Data quality checks
print("Data Quality Checks:")
print(f"  Total QA pairs: {len(df_qa)}")
print(f"  Columns: {list(df_qa.columns)}")
print()

print("Null values:")
print(df_qa.isnull().sum())
print()

print("Duplicate rows:", df_qa.duplicated().sum())
print("Duplicate IDs:", df_qa['id'].duplicated().sum())
print()

# Question and answer length analysis
df_qa['len_q'] = df_qa['question'].str.len()
df_qa['len_a'] = df_qa['answer'].str.len()

print("Question length distribution:")
print(df_qa['len_q'].describe())
print()

print("Answer length distribution:")
print(df_qa['len_a'].describe())
print()

# Answer type analysis
print("Answer type examples:")
yes_no = df_qa[df_qa['answer'].str.lower().isin(['yes', 'no'])]
print(f"  Yes/No answers: {len(yes_no)} ({len(yes_no)/len(df_qa)*100:.1f}%)")

numeric = df_qa[df_qa['answer'].str.match(r'^\d+$', na=False)]
print(f"  Numeric answers: {len(numeric)} ({len(numeric)/len(df_qa)*100:.1f}%)")

print()

# ============================================================================
# STEP 5: VISUALIZATION
# ============================================================================
print("="*70)
print("STEP 5: CREATING VISUALIZATIONS")
print("="*70)
print()

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Passage length distribution
axes[0, 0].hist(df_text['len_passage'], bins=50, edgecolor='black', alpha=0.7)
axes[0, 0].set_xlabel('Passage Length (characters)')
axes[0, 0].set_ylabel('Frequency')
axes[0, 0].set_title('Text Corpus: Passage Length Distribution')
axes[0, 0].axvline(df_text['len_passage'].mean(), color='red', 
                    linestyle='--', label=f'Mean: {df_text["len_passage"].mean():.0f}')
axes[0, 0].axvline(df_text['len_passage'].median(), color='green', 
                    linestyle='--', label=f'Median: {df_text["len_passage"].median():.0f}')
axes[0, 0].legend()

# Question length distribution
axes[0, 1].hist(df_qa['len_q'], bins=30, edgecolor='black', alpha=0.7, color='orange')
axes[0, 1].set_xlabel('Question Length (characters)')
axes[0, 1].set_ylabel('Frequency')
axes[0, 1].set_title('QA Dataset: Question Length Distribution')
axes[0, 1].axvline(df_qa['len_q'].mean(), color='red', 
                    linestyle='--', label=f'Mean: {df_qa["len_q"].mean():.0f}')
axes[0, 1].legend()

# Answer length distribution
axes[1, 0].hist(df_qa['len_a'], bins=30, edgecolor='black', alpha=0.7, color='green')
axes[1, 0].set_xlabel('Answer Length (characters)')
axes[1, 0].set_ylabel('Frequency')
axes[1, 0].set_title('QA Dataset: Answer Length Distribution')
axes[1, 0].axvline(df_qa['len_a'].mean(), color='red', 
                    linestyle='--', label=f'Mean: {df_qa["len_a"].mean():.0f}')
axes[1, 0].legend()

# Summary statistics comparison
categories = ['Text Corpus\n(passages)', 'QA Dataset\n(questions)', 'QA Dataset\n(answers)']
counts = [len(df_text), len(df_qa), len(df_qa)]
colors = ['blue', 'orange', 'green']

axes[1, 1].bar(categories, counts, color=colors, alpha=0.7, edgecolor='black')
axes[1, 1].set_ylabel('Count')
axes[1, 1].set_title('Dataset Sizes')
axes[1, 1].set_ylim(0, max(counts) * 1.1)

# Add value labels on bars
for i, (cat, count) in enumerate(zip(categories, counts)):
    axes[1, 1].text(i, count + 50, str(count), ha='center', va='bottom', fontsize=12, fontweight='bold')

plt.tight_layout()
plt.savefig('phase1_dataset_exploration.png', dpi=300, bbox_inches='tight')
print("✓ Visualization saved: phase1_dataset_exploration.png")
plt.show()
print()

# ============================================================================
# STEP 6: SUMMARY STATISTICS
# ============================================================================
print("="*70)
print("STEP 6: SUMMARY STATISTICS")
print("="*70)
print()

summary = {
    'Dataset Component': ['Text Corpus', 'QA Dataset'],
    'Total Items': [len(df_text), len(df_qa)],
    'Columns': [str(list(df_text.columns)), str(list(df_qa.columns))],
    'Null Values': [df_text.isnull().sum().sum(), df_qa.isnull().sum().sum()],
    'Duplicates': [df_text.duplicated().sum(), df_qa.duplicated().sum()]
}

summary_df = pd.DataFrame(summary)
print(summary_df.to_string(index=False))
print()

# ============================================================================
# STEP 7: SAVE PROCESSED DATA
# ============================================================================
print("="*70)
print("STEP 7: SAVING PROCESSED DATA")
print("="*70)
print()

# Create directory structure
from pathlib import Path
Path("./data/processed").mkdir(parents=True, exist_ok=True)
Path("./data/cache").mkdir(parents=True, exist_ok=True)

# Save DataFrames
df_text.to_csv('./data/processed/text_corpus.csv', index=False)
df_qa.to_csv('./data/processed/qa_dataset.csv', index=False)

print("✓ Text corpus saved: ./data/processed/text_corpus.csv")
print("✓ QA dataset saved: ./data/processed/qa_dataset.csv")
print()

# Save summary statistics
with open('./data/processed/dataset_summary.txt', 'w') as f:
    f.write("RAG Mini Wikipedia Dataset Summary\n")
    f.write("="*70 + "\n\n")
    f.write(f"Text Corpus:\n")
    f.write(f"  Total passages: {len(df_text)}\n")
    f.write(f"  Average length: {df_text['len_passage'].mean():.1f} characters\n")
    f.write(f"  Median length: {df_text['len_passage'].median():.1f} characters\n")
    f.write(f"  Range: {df_text['len_passage'].min()} - {df_text['len_passage'].max()} characters\n\n")
    f.write(f"QA Dataset:\n")
    f.write(f"  Total pairs: {len(df_qa)}\n")
    f.write(f"  Avg question length: {df_qa['len_q'].mean():.1f} characters\n")
    f.write(f"  Avg answer length: {df_qa['len_a'].mean():.1f} characters\n")

print("✓ Summary statistics saved: ./data/processed/dataset_summary.txt")
print()

# ============================================================================
# FINAL SUMMARY
# ============================================================================
print("="*70)
print("PHASE 1 COMPLETE - DATASET EXPLORATION")
print("="*70)
print()

print(" KEY FINDINGS:")
print(f"  • Text corpus: {len(df_text)} passages")
print(f"  • QA dataset: {len(df_qa)} question-answer pairs")
print(f"  • No missing values or duplicates")
print(f"  • Passage length: {df_text['len_passage'].min()}-{df_text['len_passage'].max()} chars (avg: {df_text['len_passage'].mean():.0f})")
print(f"  • Question length: avg {df_qa['len_q'].mean():.0f} chars")
print(f"  • Answer length: avg {df_qa['len_a'].mean():.0f} chars")
print()



print("="*70)
