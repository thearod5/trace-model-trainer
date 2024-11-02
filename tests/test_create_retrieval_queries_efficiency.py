import time
from unittest import TestCase

from trace_model_trainer.eval.utils import create_retrieval_queries
from trace_model_trainer.tdata.loader import load_traceability_dataset
from trace_model_trainer.utils import create_trace_map


class TestRetrievalQueriesEfficiency(TestCase):
    def test_use_case(self):
        # Load dataset
        t1, dataset = self._time_function(load_traceability_dataset, "thearod5/CCHIT")
        self.assertLess(t1, 30)

        # Create Trace Map
        t2, trace_map = self._time_function(create_trace_map, dataset.trace_df)
        self.assertLess(t2, 2)

        t3, _ = self._time_function(create_retrieval_queries, dataset)
        self.assertLess(t3, 5)

    def _step_one(self):
        ...

    def _step_two(self):
        ...

    @staticmethod
    def _time_function(func, *args, **kwargs):
        """
        A generic function to time the execution of any function.
        :param func: The function to be timed
        :param args: Positional arguments for the function
        :param kwargs: Keyword arguments for the function
        :return: Elapsed time in seconds
        """
        start_time = time.time()
        output = func(*args, **kwargs)
        end_time = time.time()
        delta = end_time - start_time
        return delta, output
