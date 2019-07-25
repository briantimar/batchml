from dataclasses import dataclass

@dataclass
class HyperParameters:
    """ Storage class for hyperparameters."""

    def __init__(self):
        pass

class Log(dict):
    """ A log which holds data from training."""

    _DEFAULT_KEYS = ['training_loss', 'hyperparameters', 'model_description']
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        if 'training_loss' not in kwargs.keys():
            self['training_loss'] = []
        if 'hyperparameters' not in kwargs.keys():
            self['hyperparameters'] = HyperParameters
        if 'model_description' not in kwargs.keys():
            self['model_description'] = "No description provided"
                
    def __getattr__(self, key):
        return self[key]
    