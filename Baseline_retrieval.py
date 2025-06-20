import numpy as np
import scipy.spatial.distance as dist
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from collections import defaultdict, Counter
import math

def retrieve_tfidf(query_tokens, vectorizer, tfidf_matrix, audio_filenames, top_k=10):
    query_vector = vectorizer.transform([query_tokens])
    scores = cosine_similarity(query_vector, tfidf_matrix).flatten()
    top_indices = np.argsort(scores)[-top_k:][::-1]
    return [(audio_filenames[i], scores[i]) for i in top_indices]

def compute_dtw_distance(query_tokens, audio_tokens):
    dist_mat = np.array([[dist.euclidean([q], [a]) for a in audio_tokens] for q in query_tokens])
    N, M = dist_mat.shape
    cost_mat = np.zeros((N + 1, M + 1))
    cost_mat[1:, 0] = np.inf
    cost_mat[0, 1:] = np.inf
    for i in range(N):
        for j in range(M):
            penalties = [cost_mat[i, j], cost_mat[i, j + 1], cost_mat[i + 1, j]]
            move = np.argmin(penalties)
            cost_mat[i + 1, j + 1] = dist_mat[i, j] + penalties[move]
    return cost_mat[N, M]

def retrieve_dtw(query_tokens, audio_sequences, audio_filenames, top_k=10):
    query_tokens = list(map(int, query_tokens))
    distances = []
    for i, seq in enumerate(audio_sequences):
        audio_tokens = list(map(int, seq.split(',')))
        dtw_distance = compute_dtw_distance(query_tokens, audio_tokens)
        distances.append((dtw_distance, i))
    distances.sort()
    return [(audio_filenames[i], d) for d, i in distances[:top_k]]

class BigTableInvertedIndex:
    def __init__(self):
        self.inverted_index = defaultdict(list)
        self.doc_freq = {}
        self.N = 0
        self.audio_filenames = []

    def build_index(self, audio_sequences, audio_filenames):
        self.N = len(audio_sequences)
        self.audio_filenames = audio_filenames
        token_doc_set = defaultdict(set)
        for idx, seq_str in enumerate(audio_sequences):
            tokens = seq_str.split(",")
            freqs = Counter(tokens)
            for token, freq in freqs.items():
                self.inverted_index[token].append((idx, freq))
                token_doc_set[token].add(idx)
        self.doc_freq = {token: len(doc_set) for token, doc_set in token_doc_set.items()}

    def compute_idf(self, token):
        df = self.doc_freq.get(token, 0)
        return math.log((self.N + 1) / (df + 1)) + 1

    def retrieve(self, query_tokens, top_k=10):
        scores = defaultdict(float)
        token_freq_in_query = Counter(query_tokens)
        for token in set(query_tokens):
            postings = self.inverted_index.get(token, [])
            idf = self.compute_idf(token)
            query_tf = token_freq_in_query[token]
            for audio_idx, audio_tf in postings:
                scores[audio_idx] += query_tf * idf * audio_tf
        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        return [(self.audio_filenames[idx], score) for idx, score in ranked[:top_k]]
