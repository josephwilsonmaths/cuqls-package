import torch
import scipy
from torch.func import jvp
torch.set_default_dtype(torch.float64)

def flatten(lst):
    tmp = [i.contiguous().view(-1, 1) for i in lst]
    return torch.cat(tmp).view(-1)

## Calibration function (ECE)
def calibration_curve_r(target,mean,variance,c):
    target = target.detach().cpu(); mean = mean.detach().cpu(); variance = variance.detach().cpu()
    predicted_conf = torch.linspace(0,1,c)
    observed_conf = torch.empty((c))
    for i,ci in enumerate(predicted_conf):
        z = scipy.stats.norm.ppf((1+ci)/2)
        ci_l = mean.reshape(-1) - z*torch.sqrt(variance.reshape(-1))
        ci_r = mean.reshape(-1) + z*torch.sqrt(variance.reshape(-1)) 
        observed_conf[i] = torch.logical_and(ci_l < target.reshape(-1), target.reshape(-1) < ci_r).type(torch.float).mean()
    return observed_conf,predicted_conf