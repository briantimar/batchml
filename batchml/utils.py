"""`utils`
    A module holding utilities for IO, and storing hyperparameters.
"""

from dataclasses import dataclass
import datetime
import os

@dataclass
class HyperParameters:
    """ Storage class for hyperparameters."""

    def __init__(self):
        pass
    
    def json(self):
        """ JSON representation of the hyperparameter settings.
        TODO"""
        return {}

class Log():
    """ A log which holds data from training."""

    _DEFAULT_KEYS = ['training_loss', 'hyperparameters', 'model_description']
    
    def __init__(self, filename=None, id='', training_loss=[],hyperparameters=None, model_description=''):
        """
        Parameters:

        `filename`: name of file where log will be written (no extension). If `None`, a default local filename based on timestamp of creation or id
        will be supplied.
        `id`: a string id to tag the log.
        `training_loss`: if not `None`, a list of scalar training losses.
        `hyperparameters`: if not `None`, an instance of `HyperParameters`.
        `model_description`: a string description of the model. """

        self.training_loss = training_loss.copy()
        if hyperparameters is None:
            self.hyperparameters = HyperParameters()
        else:
            self.hyperparameters = hyperparameters
        self.model_description = model_description

        self.id = id
        self.time_of_creation = datetime.datetime.now()
        if filename is None:
            self._filename = self._get_default_filename()
        else:
            self._filename = filename
        
        self._backend = 'json'
        if self._backend != 'json':
            raise NotImplementedError

        self._check_valid_filename()
        
    def _check_valid_filename(self):
        if '.' in self._filename:
            raise IOError("Please supply filenames without extensions.")
    
        if os.path.exists(self.filename):
            raise IOError("File {0} already exists!".format(self.filename))

    def _get_extension(self):
        """ Extension for filename."""
        extensions = {'json': '.json'}
        return extensions[self._backend]

    def _get_default_filename(self):
        """ Creates a default filename for the Log."""
        if self.id is not None:
            filename = "log_{0}".format(self.id)
        else:
            filename = self.time_of_creation.strftime("log_%Y_%m_%d__%H_%M_%S")
        return filename
        
    @property
    def filename(self):
        """Filepath, including extension, to which the log will be written."""
        return self._filename + self._get_extension()
    
    @property
    def timestamp_str(self):
        """String representation of time of creation."""
        return self.time_of_creation.strftime("%Y_%m_%d__%H_%M_%S")

    def json(self):
        """ Returns json representation of the current log state."""

        _json = dict(model_description=self.model_description,
                    time_of_creation=self.timestamp_str,
                    id=self.id,
                    hyperparameters=self.hyperparameters.json(),
                    training_loss=list( float(x) for x in self.training_loss )
                    )
        return _json


    def save(self):
        """ Save the current log state as json to specified filepath."""
        pass
