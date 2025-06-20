import numpy as np
import hnswlib
from Bio import pairwise2

def smith_waterman(seq1, seq2):
    seq1_str = " ".join(map(str, seq1))
    seq2_str = " ".join(map(str, seq2))
    score = pairwise2.align.localxx(seq1_str, seq2_str, score_only=True)
    return score

def build_hnsw_index(tfidf_matrix, dim):
    index = hnswlib.Index(space='cosine', dim=dim)
    index.init_index(max_elements=tfidf_matrix.shape[0], ef_construction=200, M=16)
    index.add_items(tfidf_matrix.toarray())
    index.set_ef(50)
    return index

def retrieve_hnsw(index, query_tokens, vectorizer, top_k=50):
    query_vector = vectorizer.transform([query_tokens]).toarray()
    labels, distances = index.knn_query(query_vector, k=top_k)
    return labels[0], distances[0]

def H_quest(query_tokens, audio_sequences, audio_filenames, top_indices, top_k=10):
    smith_scores = [smith_waterman(query_tokens, audio_sequences[idx].split(",")) for idx in top_indices]
    sorted_indices = np.argsort(smith_scores)[-top_k:][::-1]
    return [(audio_filenames[top_indices[i]], smith_scores[i]) for i in sorted_indices]
