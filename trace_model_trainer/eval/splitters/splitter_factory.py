from enum import Enum

from trace_model_trainer.eval.splitters.isplitter import ISplitter
from trace_model_trainer.eval.splitters.link_splitter import LinkSplitter
from trace_model_trainer.eval.splitters.query_splitter import QuerySplitter


class SplitterFactory(Enum):
    """Creates dataset splitters based on defined strategies."""
    QUERY = QuerySplitter
    LINK = LinkSplitter

    def create(self, *args, **kwargs) -> ISplitter:
        return self.value(*args, **kwargs)
