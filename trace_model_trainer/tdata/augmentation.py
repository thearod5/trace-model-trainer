from collections import Counter
from itertools import combinations
from typing import List


def create_augmentations():
    text_words = [w.strip().lower() for w in curr.split()]
    text_common_words = list(set(common_words).intersection(set(text_words)))

    for i in range(n_per_text):
        words_to_remove = np.random.choice(text_common_words, 3, replace=False)
        text1.append(curr)
        text2.append(remove_common_words(curr, remove_words=words_to_remove))
        labels.append(1)
    for other in artifact_texts:
        if curr == 0:
            continue
        text1.append(curr)
        text2.append(other)
        labels.append(0)


def remove_words(text: str, words: List[str]):
    """
    Removes words from text.
    :param text: The text to remove words from.
    :param words: List of words to remove
    :return: The text with words removed.
    """
    return ' '.join([w for w in text.split() if w.lower() not in words])


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


def generate_combinations(text, words, group_size: int = 3):
    """
    Generates combinations of removal of words.
    :param text: The text to remove words from.
    :param words: List of words to remove.
    :param group_size: The size of combinations to consider.
    :return: List of combinations of text with words removed.
    """
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
