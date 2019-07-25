
class TrainingInstance:
    """ A single model-training instance. Contains:
        A model
        A set of hyperparameters
        A dataset. 
       """
    
    def __init__(self, model, data_source, hyperparameters):
        from .utils import Log
        self.model = model
        self.data_source = data_source
        self.hyperparameters = hyperparameters
        self.log = Log(hyperparameters=self.hyperparameters)
    

    