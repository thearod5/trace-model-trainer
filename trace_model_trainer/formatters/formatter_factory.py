from enum import Enum

from formatters.anchor_positive_formatter import AnchorPositiveFormatter
from formatters.classification_formatter import ClassificationFormatter
from formatters.contrastive_tension_formatter import ContrastiveTensionFormatter
from formatters.iformatter import IFormatter
from formatters.triplet_formatter import TripletFormatter


class FormatterFactory(Enum):
    CLASSIFICATION = ClassificationFormatter
    ANCHOR_POSITIVE = AnchorPositiveFormatter
    CONTRASTIVE_TENSION = ContrastiveTensionFormatter
    TRIPLET = TripletFormatter

    def create(self) -> IFormatter:
        """Creates formatter for associated task."""
        return self.value()
