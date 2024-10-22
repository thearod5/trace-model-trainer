from enum import Enum

from trace_model_trainer.formatters.anchor_positive_formatter import AnchorPositiveFormatter
from trace_model_trainer.formatters.classification_formatter import ClassificationFormatter
from trace_model_trainer.formatters.contrastive_tension_formatter import ContrastiveTensionFormatter
from trace_model_trainer.formatters.iformatter import IFormatter
from trace_model_trainer.formatters.triplet_formatter import TripletFormatter


class FormatterFactory(Enum):
    CLASSIFICATION = ClassificationFormatter
    ANCHOR_POSITIVE = AnchorPositiveFormatter
    CONTRASTIVE_TENSION = ContrastiveTensionFormatter
    TRIPLET = TripletFormatter

    def create(self) -> IFormatter:
        """Creates formatter for associated task."""
        return self.value()
