import unittest
from float.data.data_loader import DataLoader
from float.pipeline.holdout_pipeline import HoldoutPipeline


class TestHoldoutPipeline(unittest.TestCase):
    def __init__(self, method_name):
        super().__init__(method_name)
        data_loader = DataLoader(file_path='../data/datasets/spambase.csv', target_col=0)
        test_set = ()
        self.holdout_pipeline = HoldoutPipeline(data_loader, test_set, 10)

    def test_init(self):
        pass

    def test_run(self):
        pass

    def test_holdout(self):
        pass

    def test_check_input(self):
        pass

    def test_start_evaluation(self):
        pass

    def test_finish_iteration(self):
        pass

    def test_finish_evaluation(self):
        pass

    def test_pretrain_predictor(self):
        pass

    def test_get_n_samples(self):
        pass

    def test_run_single_training_iteration(self):
        pass

    def test_update_progress_bar(self):
        pass

    def test_print_summary(self):
        pass
