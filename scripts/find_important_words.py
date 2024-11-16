import os
from typing import Dict, List, Tuple

import pandas as pd
from nltk import pos_tag, word_tokenize
from sentence_transformers import SentenceTransformer
from sentence_transformers.util import community_detection
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import jaccard_score

from trace_model_trainer.tdata.loader import load_traceability_dataset
from trace_model_trainer.transforms.augmentation import STOP_WORDS
from trace_model_trainer.utils import clear_memory

ORACLE = ["capture", "manage", "organize", "requirement", "requirements",
          "artifacts", "developers", "traceability",
          "trace", "link", "code",
          "impact", "change", "changes", "changed", "repositories", "reports",
          "traceability", "coverage", "team", "collaboration", "comments", "notifications", "reviews",
          "create", "edit", "categorize", "generate", "version", "gaps", "notify"]


def get_communities(sentences):
    model = SentenceTransformer("all-MiniLM-L6-v2")
    corpus_embeddings = model.encode(sentences)
    clusters = community_detection(corpus_embeddings, min_community_size=2, threshold=0.40)
    sentence_clusters = [[sentences[i] for i in cluster] for cluster in clusters]
    clear_memory(model)
    return sentence_clusters


def get_score(reference_words, predicted_words):
    all_words = list(set(reference_words + predicted_words))
    reference_vector = [1 if word in reference_words else 0 for word in all_words]
    predicted_vector = [1 if word in predicted_words else 0 for word in all_words]
    score = jaccard_score(reference_vector, predicted_vector)
    return score


def get_nouns(texts) -> Tuple[List[str], Dict[str, List[str]]]:
    ALLOWED = ["CD", "FW", "LS", "NN", "NNS", "NNP", "NNPS", "SYM", "VB", "VBD", "VBG", "VBN", "VGP", "VBZ", "PRP"]
    all_adjectives = set()
    text2words = {}

    for text in texts:
        allowed_words = [word.lower() for word, tag in pos_tag(word_tokenize(text)) if
                         tag in ALLOWED and word.lower() not in STOP_WORDS]
        all_adjectives.update(set(allowed_words))
        text2words[text] = allowed_words

    return list(set(all_adjectives)), text2words


def eval_words(method2words: Dict[str, List[str]]):
    metrics = []
    for method_name, words in method2words.items():
        captured = set(ORACLE).intersection(set(words))
        missed = set(ORACLE).difference(set(words))
        extra = set(words).difference(set(ORACLE))

        method_metric = {
            "method": method_name,
            'captured': len(captured) / len(ORACLE),
            'missed': len(missed) / len(ORACLE),
            'extra': len(extra) / len(words),
            "score": get_score(ORACLE, words)
        }
        metrics.append(method_metric)
        print(method_name)
        print("Missed:", pos_tag(list(missed)))
    print(pd.DataFrame(metrics))


def runner(dataset_path: str):
    dataset = load_traceability_dataset(dataset_path)
    texts = list(dataset.artifact_map.values())

    important_words0 = get_nouns(texts)
    # Calculate stop words via VSM
    vectorizer = TfidfVectorizer()
    vectorizer.fit(texts)
    sorted_word_scores = sorted(zip(vectorizer.get_feature_names_out(), vectorizer.idf_), key=lambda b: b[1])
    stop_words = [word for word, score in sorted_word_scores if score < 0.5]
    important_words1 = [w for w in important_words0 if w not in stop_words]

    eval_words({
        "nltk": important_words0,
        "nltk+clustering": important_words1,
        # "intersection": list(set(important_words0).intersection(set(important_words1)))
    })


if __name__ == '__main__':
    runner(os.path.expanduser("~/projects/trace-model-trainer/res/test"))
