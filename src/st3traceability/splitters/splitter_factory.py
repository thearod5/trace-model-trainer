from enum import Enum

from st3traceability.splitters.isplitter import ISplitter
from st3traceability.splitters.query_splitter import QuerySplitter


class SplitterFactory(Enum):
    """Creates dataset splitters based on defined strategies."""
    QUERY = QuerySplitter

    def create(self, *args, **kwargs) -> ISplitter:
        return self.value(*args, **kwargs)
