from enum import Enum

from st3traceability.formatters.anchor_positive_formatter import AnchorPositiveFormatter
from st3traceability.formatters.classification_formatter import ClassificationFormatter
from st3traceability.formatters.iformatter import IFormatter


class FormatterFactory(Enum):
    CLASSIFICATION = ClassificationFormatter
    ANCHOR_POSITIVE = AnchorPositiveFormatter

    def create(self) -> IFormatter:
        """Creates formatter for associated task."""
        return self.value()
