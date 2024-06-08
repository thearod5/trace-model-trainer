import spacy


class ArtifactProcessor:
    def __init__(self):
        # Load the pre-trained spaCy model
        self.nlp = spacy.load("en_core_web_sm")

    def extract_entities_and_actions(self, text):
        # Process the text using spaCy
        n_words = len(text.split())
        if n_words < 5:
            return text

        doc = self.nlp(text)

        entities_and_actions = []

        for ent in doc.ents:
            entities_and_actions.append((ent.start, ent.text))

        for token in doc:
            if token.pos_ == "VERB":
                entities_and_actions.append((token.i, token.lemma_))

        # Sort by the original order in the text
        entities_and_actions.sort()

        # Extract the words, preserving the order
        extracted_text = ' '.join([item[1] for item in entities_and_actions])

        return extracted_text
