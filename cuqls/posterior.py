import torch
import cuqls.regression
import cuqls.classification

def Cuqls(
    network: torch.nn.Module,
    task: str ='regression'
    ):
    
    if task == 'regression':
        return cuqls.regression.regressionParallel(network)
    
    elif task == 'classification':
        return cuqls.classification.classificationParallel(network)
    
    elif task == 'classificationInterpolation':
        return cuqls.classification.classificationParallelInterpolation(network)
    
