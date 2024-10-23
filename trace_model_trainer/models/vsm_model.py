import string
from typing import Dict, List

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from trace_model_trainer.models.itrace_model import ITraceModel, SimilarityMatrix
from trace_model_trainer.tdata.trace_dataset import TraceDataset
from trace_model_trainer.tdata.types import TracePrediction

prepositions = {
    'aboard', 'about', 'above', 'across', 'after', 'against', 'along', 'amid', 'among', 'anti', 'around', 'as', 'at',
    'before', 'behind', 'below', 'beneath', 'beside', 'besides', 'between', 'beyond', 'but', 'by', 'concerning',
    'considering', 'despite', 'down', 'during', 'except', 'excepting', 'excluding', 'following', 'for', 'from',
    'in', 'inside', 'into', 'like', 'minus', 'near', 'of', 'off', 'on', 'onto', 'opposite', 'outside', 'over',
    'past', 'per', 'plus', 'regarding', 'round', 'save', 'since', 'than', 'through', 'to', 'toward', 'towards',
    'under', 'underneath', 'unlike', 'until', 'up', 'upon', 'versus', 'via', 'with', 'within', 'without'
}


class VSMModel(ITraceModel):
    def __init__(self):
        self.vectorizer = TfidfVectorizer(ngram_range=(1, 2))
        self.has_trained = False

    def train(self, dataset_map: TraceDataset | Dict[str, TraceDataset]):
        """
        Trains VSM model on training dataset.
        :param dataset_map: The dataset
        :return: None.
        """
        if not isinstance(dataset_map, dict):
            dataset_map = {"train_dataset": dataset_map}
        texts = []
        for _, dataset in dataset_map.items():
            texts.extend(dataset.artifact_map.values())
        processed_content = [self.preprocess(t) for t in texts]
        self.vectorizer.fit(processed_content)
        self.has_trained = True

    def predict(self, sources: List[str], targets: List[str]) -> SimilarityMatrix:
        """
        Predicts similarity scores between sources and targets using VSM.
        :param sources: The list of source texts.
        :param targets: The list of target texts.
        :return: list of trace predictions.
        """
        if not self.has_trained:
            raise Exception("VSM model has not been trained.")
        # Transform sources and targets
        source_matrix = self.vectorizer.transform(sources)
        target_matrix = self.vectorizer.transform(targets)
        # Compute cosine similarity
        similarity_matrix = cosine_similarity(source_matrix, target_matrix)
        return similarity_matrix

    def predict_single(self, source: str, target: str) -> TracePrediction:
        """
        Predicts similarity score between source and target using VSM.
        :param source: The source text.
        :param target: The target text.
        :return: trace prediction.
        """
        return self.predict([source], [target])[0]

    @staticmethod
    def preprocess(text):
        # Convert to lowercase
        text = text.lower()
        # Remove punctuation
        text = text.translate(str.maketrans('', '', string.punctuation))
        # Remove extra whitespace
        text = text.strip()
        return text

    def get_top_n_words(self, text, n, threshold: float):
        if not self.has_trained:
            raise Exception("This model is not trained yet.")
        words = self.preprocess(text).split()
        if len(words) <= n:
            return text
        word2score = {word: self.get_score(word.lower()) for word in words}
        selected_words = [w for w in words if word2score[w] >= threshold and w not in prepositions]
        if len(selected_words) == 0:
            selected_words = [word2score[max(word2score.values())]]
        transformed = ' '.join(selected_words)
        return transformed

    def get_score(self, word: str):
        word_idx = self.vectorizer.vocabulary_.get(word, -1)
        if word_idx == -1:
            return 0
        score = self.vectorizer.idf_[word_idx]
        return score
