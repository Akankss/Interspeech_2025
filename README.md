# QbE-STD Retrieval Pipeline

This repository contains code for H-QuEST: Accelerating Query-by-Example Spoken Term Detection with Hierarchical Indexing featuring:

- Feature extraction from audio using Wav2Vec2 (`feature_extraction.py`)
- HQuEST retrieval with HNSW + Smith-Waterman (`hquest_retrieval.py`)
- Baseline retrieval methods including TF-IDF, DTW, and a BigTable-style inverted index (`baseline_retrievals.py`)
- Main pipeline to run retrieval and output results (`main.py`)

## Usage
Extract Features
python feature_extraction.py --input_dir path/to/audio --output_csv features.csv

Run Retrieval 
python main.py --audio_csv features.csv --query_csv queries.csv --output_csv results.csv --top_k 10




