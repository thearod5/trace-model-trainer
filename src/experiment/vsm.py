import string
from typing import List

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

prepositions = {
    'aboard', 'about', 'above', 'across', 'after', 'against', 'along', 'amid', 'among', 'anti', 'around', 'as', 'at',
    'before', 'behind', 'below', 'beneath', 'beside', 'besides', 'between', 'beyond', 'but', 'by', 'concerning',
    'considering', 'despite', 'down', 'during', 'except', 'excepting', 'excluding', 'following', 'for', 'from',
    'in', 'inside', 'into', 'like', 'minus', 'near', 'of', 'off', 'on', 'onto', 'opposite', 'outside', 'over',
    'past', 'per', 'plus', 'regarding', 'round', 'save', 'since', 'than', 'through', 'to', 'toward', 'towards',
    'under', 'underneath', 'unlike', 'until', 'up', 'upon', 'versus', 'via', 'with', 'within', 'without'
}


class VSMController:
    def __init__(self):
        self.vectorizer = TfidfVectorizer(ngram_range=(1, 2))
        self.has_trained = False

    def train(self, texts: List[str]):
        processed_content = [self.preprocess(t) for t in texts]
        self.vectorizer.fit(processed_content)
        self.has_trained = True

    def preprocess(self, text):
        # Convert to lowercase
        text = text.lower()
        # Remove punctuation
        text = text.translate(str.maketrans('', '', string.punctuation))
        # Remove extra whitespace
        text = text.strip()
        return text

    def predict(self, sources: List[str], targets: List[str]):
        if not self.has_trained:
            all_content = sources + targets
            self.train(all_content)
        # Transform sources and targets
        source_matrix = self.vectorizer.transform(sources)
        target_matrix = self.vectorizer.transform(targets)
        # Compute cosine similarity
        similarity_matrix = cosine_similarity(source_matrix, target_matrix)
        return similarity_matrix

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
