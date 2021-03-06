import unittest


class TestRunner:
    """
    Class to run all unit tests in the tests directory. Must be run from within the unit_test directory.
    """
    def __init__(self):
        self.loader = unittest.TestLoader()
        self.start_dir = './'

    def run(self):
        """
        Run all unit tests in the tests directory.
        """
        suite = self.loader.discover(self.start_dir)
        runner = unittest.TextTestRunner()
        runner.run(suite)
