import torch
import nuqls.regression
import nuqls.regressionfull
import nuqls.classification

def Nuqls(
    network: torch.nn.Module,
    task: str ='regression'
    ):
    
    if task == 'regression':
        return nuqls.regression.regressionParallel(network)
    
    elif task == 'classification':
        return nuqls.classification.classificationParallel(network)
    
