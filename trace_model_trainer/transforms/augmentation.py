import re
from collections import defaultdict
from itertools import combinations
from typing import List

import numpy as np
import pandas as pd
from datasets import Dataset
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm

from trace_model_trainer.utils import clear_memory

STOP_WORDS = ["of", "a", "and", "so", "on", "to", "at", "by", "for", "in", "with", "is", "them", "has", "like", "allow", "be", "able",
              "when", "that", "the", "been", "through"]
GENERATIONS_PER_SAMPLE = 3
DIRTY_PER_SAMPLE = 1
np.random.seed(1600)


def create_augmented_dataset(texts: List[str]):
    aug_methods = [
        "identity",
        "important",
        "dirty"
        # "ngrams"
    ]  # []  # ["dirty"] # ["important"] # ["dirty", "important"]
    print("Creating augmented dataset")

    text2phrase = get_tfidf_important_phrase(texts)

    text1 = []
    text2 = []
    labels = []
    n_pos = 0

    # Add ngram links
    if "ngrams" in aug_methods:
        ngram_links = generate_ngram_links(texts)
        for linked_texts in ngram_links:
            for a_text, b_text in combinations(linked_texts, 2):
                text1.append(a_text)
                text2.append(b_text)
                labels.append(1)

    for text in tqdm(texts, desc="Augmenting dataset samples"):
        # Add identity
        if "identity" in aug_methods:
            text1.append(text)
            text2.append(text)
            labels.append(1)
            n_pos += 1

        if "important" in aug_methods:
            text_phrase, text_important_words = text2phrase[text]

            text1.append(text)
            text2.append(text_phrase)
            labels.append(1)

            # text_important_words = np.random.choice(text_important_words, min(GENERATIONS_PER_SAMPLE, len(text_important_words)))
            # for a_text in text_important_words:
            #     text1.append(text)
            #     text2.append(a_text)
            #     labels.append(.75)
            #     n_pos += 1

        if "dirty" in aug_methods:
            text_phrase, text_important_words = text2phrase[text]
            text_common_words = [w for w in split(text) if w.lower() not in text_important_words]
            text_common_words = np.random.choice(text_common_words, 3)
            for common_word in text_common_words:
                text1.append(text)
                text2.append(common_word)
                labels.append(0.1)

        # Sampler negatives for text
        negative_texts = get_negatives(text, texts, int(len(text) * .10))
        for other in negative_texts:
            text1.append(text)
            text2.append(other)
            labels.append(0)

        a = 1
    print("Training Data:\n", pd.Series(labels).value_counts())

    return Dataset.from_dict({
        "text1": text2,
        "text2": text1,
        "label": labels
    })


def create_text2important_words(texts: List[str], k: int = 20):
    common_words = get_common_words(texts)
    text2words = defaultdict(list)
    for t in texts:
        t_important_words = [w for w in split(t) if w.lower() not in common_words and w.lower() not in STOP_WORDS]
        text2words[t] = t_important_words
    return text2words


def get_tfidf_important_phrase(texts: List[str]):
    # TF-IDF Analysis
    vectorizer = TfidfVectorizer(stop_words=STOP_WORDS)
    X = vectorizer.fit_transform(texts)

    # Extract the vocabulary and the TF-IDF matrix
    corpus_vocabulary = np.array(vectorizer.get_feature_names_out())
    text2words = {}

    # Iterate through each document's row in the sparse matrix
    for row_index in range(X.shape[0]):
        # Get the non-zero elements in the row
        row = X[row_index]
        non_zero_indices = row.nonzero()[1]
        word_scores = zip(non_zero_indices, row.data)

        # Sort word scores in descending order
        sorted_word_scores = sorted(word_scores, key=lambda x: x[1], reverse=True)

        # Extract phrase containing important words
        # TODO: Instead of fixed numbers, find outliers and include those
        text = texts[row_index]
        text_id = text.lower()
        important_words = [corpus_vocabulary[idx] for idx, _ in sorted_word_scores[:5]]
        word_indices = [text_id.index(word) for word in important_words[:3]]

        start_idx = min(word_indices)
        end_idx = max(word_indices)

        text2words[text] = (text[start_idx: end_idx], important_words)

    return text2words


def get_negatives(text: str, candidates: List[str], n_items: int = None):
    text_words = set(split(text))
    candidate2intersection = {c: set(split(c)).intersection(text_words) for c in candidates}
    sorted_candidates = sorted(candidate2intersection.items(), key=lambda t: len(t[1]), reverse=False)
    return [c[0] for c in sorted_candidates[:n_items]]


def remove_words(text: str, words: List[str]):
    """
    Removes words from text.
    :param text: The text to remove words from.
    :param words: List of words to remove
    :return: The text with words removed.
    """
    return ' '.join([w for w in text.split() if w.lower() not in words])


def generate_dirty_combinations(text, words, group_size: int = None):
    """
    Generates combinations of removal of words.
    :param text: The text to remove words from.
    :param words: List of words to remove.
    :param group_size: The size of combinations to consider. Defaults to length of words - 1.
    :return: List of combinations of text with words removed.
    """
    if group_size is None:
        group_size = len(words) - 1
    # Split the text into individual words
    text_words = text.split()

    # Generate all combinations of the words list
    word_combinations = list(combinations(words, group_size))  # Create pairs of words to remove

    # Create the resulting texts with the words removed
    resulting_texts = []
    for combo in word_combinations:
        modified_text = [word for word in text_words if word not in combo]
        resulting_texts.append(" ".join(modified_text))

    return resulting_texts


def split(text: str):
    return re.findall(r'\b\w+\b', text.lower())


#
# RETIRED
#

def get_top_words(texts: List[str], top_n: int = 30):
    """
    Calculates the top n words in texts.
    :param texts: The texts to analyze.
    :param top_n: The number of top words to return.
    :return: List of top words used in texts.
    """
    word2count = {}
    for a_text in texts:
        for word in a_text.split():
            word_id = word.lower()
            if word_id not in word2count:
                word2count[word_id] = 0
            word2count[word_id] += 1

    common_word_map = list(sorted(word2count.items(), key=lambda t: t[1], reverse=True))[:top_n]
    common_words = [w[0] for w in common_word_map]
    return common_words


def get_common_words(texts: List[str]):
    # TF-IDF Analysis
    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(texts)

    corpus_vocabulary = vectorizer.get_feature_names_out()
    text2words = {}
    for row_index in range(X.shape[0]):
        text = texts[row_index]
        text_word2score = {word: X[row_index, i] for i, word in enumerate(corpus_vocabulary)}
        text_word2score = {word: word_score for word, word_score in sorted(text_word2score.items(), key=lambda b: b[1], reverse=True)
                           if word_score > 0}
        text2words[text] = text_word2score
    idf_scores = dict(zip(vectorizer.get_feature_names_out(), vectorizer.idf_))

    # Calculate the bottom quartile (25th percentile) of IDF scores
    idf_values = np.array(list(idf_scores.values()))
    bottom_quartile_threshold = np.percentile(idf_values, 10)

    # calculate important words
    text2words = defaultdict(list)
    for text in texts:
        word_score_map = {w: idf_scores.get(w, 0) for w in split(text)}
        words_sorted = sorted(word_score_map.items(), key=lambda w, s: s, reverse=True)
        text2words[text] = words_sorted[:5]
    # Identify potential stop words (words with low IDF)
    stop_words_candidates = [word for word, score in idf_scores.items() if score + .01 <= bottom_quartile_threshold]
    return stop_words_candidates


def generate_ngram_links(texts: List[str], n: int = 2):
    """
    Finds n-grams shared across texts.
    :param texts: The texts to analyze for n-grams.
    :param n: The number of words to string tog
    :return:
    """
    all_ngrams = defaultdict(list)
    for sentence in texts:
        tokens = split(sentence)
        ngrams = generate_ngrams(tokens, n)
        for ngram in ngrams:
            if any([word in STOP_WORDS for word in ngrams]):
                continue
            all_ngrams[ngram].append(sentence)
    all_ngrams = {k: v for k, v in all_ngrams.items() if len(v) > 1}
    ngram_links = list(all_ngrams.values())
    return ngram_links


def generate_ngrams(tokens, n):
    return zip(*[tokens[i:] for i in range(n)])


def calculate_hard_negatives(texts: List[str]):
    model = SentenceTransformer("all-MiniLM-L6-v2")
    embeddings = model.encode(texts)
    similarity_matrix = cosine_similarity(embeddings, embeddings)

    text2negatives = {}

    for i, text in enumerate(texts):
        negatives = [b[1] for b in sorted(zip(similarity_matrix[i], texts), key=lambda t: t[0], reverse=True)]
        text2negatives[text] = negatives

    clear_memory(model)
    return text2negatives
