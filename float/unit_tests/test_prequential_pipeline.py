import unittest
from float.data.data_loader import DataLoader
from float.pipeline.prequential_pipeline import PrequentialPipeline


class TestPrequentialPipeline(unittest.TestCase):
    def __init__(self, method_name):
        super().__init__(method_name)
        data_loader = DataLoader(path='../data/datasets/spambase.csv', target_col=-1)
        self.prequential_pipeline = PrequentialPipeline(data_loader)

    def test_init(self):
        pass

    def test_run(self):
        pass

    def test_run_prequential(self):
        pass

    def test_validate(self):
        pass

    def test_start_evaluation(self):
        pass

    def test_pretrain_predictor(self):
        pass

    def test_run_iteration(self):
        pass

    def test_get_n_batch(self):
        pass

    def test_finish_iteration(self):
        pass

    def test_update_progress_bar(self):
        pass

    def test_finish_evaluation(self):
        pass

    def test_print_summary(self):
        pass
