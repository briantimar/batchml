import unittest

class LogTestCase(unittest.TestCase):

    def test_log_init(self):
        """ Check that log has required params."""
        from .utils import Log
        l = Log()
        self.assertTrue(hasattr(l, 'training_loss'))
        self.assertTrue(hasattr(l, 'hyperparameters'))
        self.assertTrue(hasattr(l, 'model_description'))

    def test_filename(self):
        from .utils import Log
        log = Log(filename="test")
        self.assertEqual(log.filename, "test.json")

    def test_backend(self):
        from .utils import Log
        l = Log()
        self.assertEqual(l._backend, 'json')

class TrainingInstanceTestCase(unittest.TestCase):

    def test_log_existence(self):
        """ Check that logs are created upon TrainingInstance creation."""
        from .training import TrainingInstance
        instance = TrainingInstance(None, None, {})
        self.assertTrue(hasattr(instance, 'log'))


if __name__ == "__main__":
    unittest.main()