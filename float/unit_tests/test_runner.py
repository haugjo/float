import unittest


class TestRunner:
    """
    Class to run all unit tests in the unit_tests directory.
    """
    def __init__(self):
        loader = unittest.TestLoader()
        start_dir = '../unit_tests'
        self.suite = loader.discover(start_dir)
        self.runner = unittest.TextTestRunner()

    def run(self):
        """
        Run all unit tests in the unit_tests directory.
        """
        self.runner.run(self.suite)
