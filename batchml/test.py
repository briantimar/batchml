import unittest
import os
import json
from .utils import Log, HyperParameters

class LogTestCase(unittest.TestCase):

    def test_log_init(self):
        """ Check that log has required params."""
        l = Log()
        self.assertTrue(hasattr(l, 'training_loss'))
        self.assertTrue(hasattr(l, 'hyperparameters'))
        self.assertTrue(hasattr(l, 'model_description'))

    def test_filename(self):
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
        l = Log()
        self.assertEqual(l._backend, 'json')

    def test_json(self):
        import numpy as np

        l = Log(id='id', training_loss=np.asarray([2.3], dtype=np.float32))
        j = l.json()
        for key in ('model_description', 'training_loss', 'hyperparameters'):
            self.assertTrue(key in j.keys())
        with open("test.json", 'w') as f:
            json.dump(j,f)
        os.remove("test.json")
        
    def test_save(self):
        l = Log(id='id', filename="test")
        l.save()
        with open(l.filename) as f:
            j = json.load(f)
            self.assertEqual(j['id'], 'id')
        os.remove("test.json")

    def test_from_json(self):
        import datetime
        jsn = dict(id='id', model_description='model', training_loss=[2],
                    hyperparameters={'batch_size': 0, 'epochs': 1, 'learning_rate':0.0}, 
                   time_of_creation="2019_07_25__13_13_13")
        
        log = Log.from_json(jsn, filename='test')
        self.assertEqual(log.id, 'id')
        self.assertEqual(log.training_loss, [2])
        self.assertEqual(log.time_of_creation, 
                            datetime.datetime(year=2019,month=7,day=25,hour=13,minute=13,second=13))
        self.assertEqual(log.hyperparameters.epochs, 1)

    def test_load(self):
        l = Log(filename="test", id='id')
        l.save()
        l2 = Log.load("test.json")
        os.remove("test.json")
        self.assertTrue(l.id==l2.id)
        self.assertTrue(l.timestamp_str == l2.timestamp_str)

class HyperParametersTestCase(unittest.TestCase):

    def test_from_json(self):
        jsn = { 
            'batch_size': 32, 
            'epochs': 12, 
            'learning_rate': .01
        }
        hp = HyperParameters.from_json(jsn)
        self.assertEqual(hp.epochs, jsn['epochs'])

    def test_json(self):
        hp = HyperParameters(learning_rate=.03)
        jsn = hp.json()
        self.assertEqual(jsn['learning_rate'], .03)

class TrainingInstanceTestCase(unittest.TestCase):

    def test_log_existence(self):
        """ Check that logs are created upon TrainingInstance creation."""
        from .training import TrainingInstance
        instance = TrainingInstance(None, None, {})
        self.assertTrue(hasattr(instance, 'log'))


if __name__ == "__main__":
    unittest.main()
