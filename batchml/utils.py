"""`utils`
    A module holding utilities for IO, and storing hyperparameters.
"""

from dataclasses import dataclass
import datetime

@dataclass
class HyperParameters:
    """ Storage class for hyperparameters."""

    def __init__(self):
        pass

class Log(dict):
    """ A log which holds data from training."""

    _DEFAULT_KEYS = ['training_loss', 'hyperparameters', 'model_description']
    
    def __init__(self, filename=None, id=None, **kwargs):
        """
        Parameters:

        `filename`: name of file where log will be written. If None, a default local filename based on timestamp of creation or id
        will be supplied.
        `id`: if not None, an id to tag the log. """
        super().__init__(**kwargs)
        if 'training_loss' not in kwargs.keys():
            self['training_loss'] = []
        if 'hyperparameters' not in kwargs.keys():
            self['hyperparameters'] = HyperParameters
        if 'model_description' not in kwargs.keys():
            self['model_description'] = "No description provided"

        self.id = id
        self.time_of_creation = datetime.datetime.now()
        if 'filename' not in kwargs.keys():
            self._filename = self._get_default_filename()
        else:
            self._filename = kwargs['filename']

        self._check_valid_filename(self._filename)
        self._backend = 'json'

        if self._backend != 'json':
            raise NotImplementedError

    def __getattr__(self, key):
        return self[key]
    
    def _get_extension(self):
        """ Extension for filename."""
        extensions = {'json': '.json'}
        return extensions[self._backend]

    def _get_default_filename(self):
        """ Creates a default filename for the Log."""
        if self.id is not None:
            putative_filename = "log_{0}".format(self.id)
        else:
            putative_filename = self.time_of_creation.strftime("log_%Y_%m_%d__%H_%M_%S")
        

    @property
    def filename(self):
        """Filepath to which the log will be written."""
        return self._filename

    def save(self):
        """ Save the current log state as json to specified filepath."""
        pass
