import unittest

class LogTestCase(unittest.TestCase):

    def test_log_init(self):
        """ Check that log has required params."""
        from .utils import Log
        l = Log()
        self.assertTrue(hasattr(l, 'training_loss'))
        self.assertTrue(hasattr(l, 'hyperparameters'))
        self.assertTrue(hasattr(l, 'model_description'))

class TrainingInstanceTestCase(unittest.TestCase):

    def test_log_existence(self):
        """ Check that logs are created upon TrainingInstance creation."""
        from .training import TrainingInstance
        instance = TrainingInstance(None, None, {})
        self.assertTrue(hasattr(instance, 'log'))


if __name__ == "__main__":
    unittest.main()