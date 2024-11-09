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

    text2important_words = get_tfidf_important_words(texts)

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

    for text in texts:
        # Add identity
        if "identity" in aug_methods:
            text1.append(text)
            text2.append(text)
            labels.append(1)
            n_pos += 1

        if "important" in aug_methods:
            text_important_words = text2important_words[text]
            text_important_words = np.random.choice(text_important_words, min(GENERATIONS_PER_SAMPLE, len(text_important_words)))
            if len(text_important_words) > 0:
                for a_text in text_important_words:
                    text1.append(text)
                    text2.append(a_text)
                    labels.append(.75)
                    n_pos += 1

        if "dirty" in aug_methods:
            text_important_words = text2important_words[text]
            text_common_words = [w for w in split(text) if w.lower() not in text_important_words]
            text_common_words = text_common_words[:10]
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


def get_tfidf_important_words(texts: List[str]):
    # TF-IDF Analysis
    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(texts)

    corpus_vocabulary = vectorizer.get_feature_names_out()
    text2words = {}
    for row_index in range(X.shape[0]):
        text_word2score = {word: X[row_index, i] for i, word in enumerate(corpus_vocabulary)}
        text_word2score = {word: word_score for word, word_score in sorted(text_word2score.items(), key=lambda b: b[1], reverse=True)
                           if word_score > 0}
        n_keep = len(text_word2score) // 2
        text_word2score = [word for word, word_score in list(text_word2score.items())[:n_keep] if word not in STOP_WORDS]
        text2words[texts[row_index]] = text_word2score
    return text2words


def create_word2text(texts: List[str], most_common_first: bool = True):
    word2text = defaultdict(list)
    for t in texts:
        for t_word in split(t):
            word2text[t_word.lower()].append(t)
    sorted_word2text = {k: v for k, v in  # most used to least used
                        sorted(word2text.items(), reverse=most_common_first, key=lambda w2t: len(w2t[1]))}
    return sorted_word2text


def get_negatives(text: str, candidates: List[str], n_items: int = None):
    text_words = set(split(text))
    candidate2intersection = {c: set(split(c)).intersection(text_words) for c in candidates}
    sorted_candidates = sorted(candidate2intersection.items(), key=lambda t: len(t[1]), reverse=False)
    return [c[0] for c in sorted_candidates[:n_items]]


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


def split(text: str):
    return re.findall(r'\b\w+\b', text.lower())
