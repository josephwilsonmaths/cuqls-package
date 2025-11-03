import torch
import cuqls.regression
import cuqls.classification

def Cuqls(
    network: torch.nn.Module,
    task: str ='regression',
    precision: str = 'double'
    ):
    
    if task == 'regression':
        return cuqls.regression.regressionParallel(network)
    
    elif task == 'classification':
        return cuqls.classification.classificationParallel(network, precision)
    
    elif task == 'classificationInterpolation':
        return cuqls.classification.classificationParallelInterpolation(network, precision)
    
