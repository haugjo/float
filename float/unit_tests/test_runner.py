import unittest


class TestRunner:
    """
    Class to run all unit tests in the unit_tests directory.
    """
    def __init__(self):
        self.loader = unittest.TestLoader()
        self.start_dir = './'

    def run(self):
        """
        Run all unit tests in the unit_tests directory.
        """
        suite = self.loader.discover(self.start_dir)
        runner = unittest.TextTestRunner()
        runner.run(suite)
