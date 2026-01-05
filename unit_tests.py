from test_folder import *
import unittest

if __name__ == "__main__":
  loader = unittest.TestLoader()
  tests = loader.discover('./test_folder')
  testRunner = unittest.runner.TextTestRunner()
  testRunner.run(tests)
