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
        import os

        log = Log(filename="test")
        self.assertEqual(log.filename, "test.json")

        with open("test.json", 'w') as f:
            f.write("{}")
        with self.assertRaises(IOError):
            log = Log(filename="test")
        os.remove("test.json")
        with self.assertRaises(IOError):
            log = Log(filename="test.json")

    def test_backend(self):
        from .utils import Log
        l = Log()
        self.assertEqual(l._backend, 'json')

    def test_json(self):
        from .utils import Log
        import json
        import os
        import numpy as np

        l = Log(id='id', training_loss=np.asarray([2.3], dtype=np.float32))
        j = l.json()
        for key in ('model_description', 'training_loss', 'hyperparameters'):
            self.assertTrue(key in j.keys())
        with open("test.json", 'w') as f:
            json.dump(j,f)
        os.remove("test.json")
        


class TrainingInstanceTestCase(unittest.TestCase):

    def test_log_existence(self):
        """ Check that logs are created upon TrainingInstance creation."""
        from .training import TrainingInstance
        instance = TrainingInstance(None, None, {})
        self.assertTrue(hasattr(instance, 'log'))


if __name__ == "__main__":
    unittest.main()