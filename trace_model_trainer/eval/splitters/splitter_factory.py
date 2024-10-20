from enum import Enum

from eval.splitters.isplitter import ISplitter
from eval.splitters.link_splitter import LinkSplitter
from eval.splitters.query_splitter import QuerySplitter


class SplitterFactory(Enum):
    """Creates dataset splitters based on defined strategies."""
    QUERY = QuerySplitter
    LINK = LinkSplitter

    def create(self, *args, **kwargs) -> ISplitter:
        return self.value(*args, **kwargs)
