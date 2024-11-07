from collections import Counter
from itertools import combinations
from typing import List

import numpy as np
from datasets import Dataset
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

from trace_model_trainer.utils import clear_memory


def create_augmented_dataset(texts: List[str]):
    aug_methods = ["important", "dirty"]  # []  # ["dirty"] # ["important"] # ["dirty", "important"]
    print("Creating augmented dataset")
    n_pos = 1

    top_words = get_top_words(texts, 20)
    top_words_set = set(top_words)

    text1 = []
    text2 = []
    labels = []

    for text in texts:
        # Create positive examples
        n_added = 0
        text_words_set = set(text.split())
        text_common_words = list(top_words_set.intersection(text_words_set))
        text_important_words = list(text_words_set.difference(top_words_set))

        # Add identity
        text1.append(text)
        text2.append(text)
        labels.append(1)
        n_added += 1

        # Add dirty identity combinations
        if "dirty" in aug_methods:
            dirty_combinations = generate_combinations(text, text_common_words)
            if len(dirty_combinations) == 0:
                raise Exception("No dirty combinations found")

            selected_dirty = np.random.choice(dirty_combinations, size=n_pos)
            for dirty in selected_dirty:
                text1.append(text)
                text2.append(dirty)
                labels.append(1)
                n_added += 1

        # Add important words
        if "important" in aug_methods:
            selected_augmentations = np.random.choice(text_important_words, size=n_pos)
            for a_text in selected_augmentations:
                text1.append(text)
                text2.append(a_text)
                labels.append(1)
                n_added += 1

        # Generate equal number of negative examples
        negative_texts = np.random.choice(texts, size=(n_added * 2))
        for other in negative_texts:
            if text == 0:
                continue
            text1.append(text)
            text2.append(other)
            labels.append(0)

    return Dataset.from_dict({
        "text1": text2,
        "text2": text1,
        "label": labels
    })


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


def generate_combinations(text, words, group_size: int = None):
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


def generate_ngraphs(sentences: List[str]):
    n = 2  # For bigrams; change to 3 for trigrams, etc.
    all_ngrams = []
    for sentence in sentences:
        tokens = sentence.split()
        ngrams = generate_ngrams(tokens, n)
        all_ngrams.extend(ngrams)
    ngram_counts = Counter(all_ngrams)
    most_common_ngrams = ngram_counts.most_common()

    return most_common_ngrams


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
