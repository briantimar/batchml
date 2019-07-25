import unittest

class UtilsTestCase(unittest.TestCase):

    def test_log(self):
        from .utils import Log
        l = Log()
        self.assertTrue(hasattr(l, 'training_loss'))
        self.assertTrue(hasattr(l, 'hyperparameters'))
        self.assertTrue(hasattr(l, 'model_description'))

if __name__ == "__main__":
    unittest.main()