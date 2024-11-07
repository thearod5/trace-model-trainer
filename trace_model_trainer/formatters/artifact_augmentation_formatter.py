from datasets import Dataset

from trace_model_trainer.formatters.iformatter import IFormatter
from trace_model_trainer.tdata.augmentation import generate_combinations, get_top_words
from trace_model_trainer.tdata.trace_dataset import TraceDataset


class ArtifactAugmentationFormatter(IFormatter):

    def format(self, dataset: TraceDataset) -> Dataset:
        artifact_texts = list(dataset.artifact_df["content"])

        # Calculate top words in tets
        common_words = get_top_words(artifact_texts)

        text1 = []
        text2 = []
        labels = []

        for curr_text in artifact_texts:
            # Generate Positive Augmentations
            text_words = [w.strip().lower() for w in curr_text.split()]
            text_common_words = list(set(common_words).intersection(set(text_words)))
            augmented_texts = generate_combinations(curr_text, text_common_words, len(text_common_words) - 1)
            for a_text in augmented_texts:
                text1.append(curr_text)
                text2.append(a_text)
                labels.append(1)

            # Generate negative labels
            for other in artifact_texts:
                if curr_text == 0:
                    continue
                text1.append(curr_text)
                text2.append(other)
                labels.append(0)

        # Step 5: Format the dataset
        return Dataset.from_dict({
            "text1": text1,
            "text2": text2,
            "label": labels
        })
