import unittest

loader = unittest.TestLoader()
start_dir = '../unit_tests'
suite = loader.discover(start_dir)

runner = unittest.TextTestRunner()
runner.run(suite)
