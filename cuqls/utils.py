import torch
import scipy
import numpy as np
import sklearn.metrics as sk
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

def ternary_search(f, left, right, its, verbose=False, input_name = 'input', output_name = 'function'):
    left_input, right_input = left, right

    # Ternary Search
    for _ in range(its):
        left_third = left_input + (right_input - left_input) / 3
        right_third = right_input - (right_input - left_input) / 3

        # Left function value
        left_f = f(left_third)

        # Right ECE
        right_f = f(right_third)

        if left_f > right_f:
            left_input = left_third
        else:
            right_input = right_third

        input = (left_input + right_input) / 2

        # Print info
        if verbose:
            print(f'\n{input_name}: {input:.3}; {output_name}: [{left_f:.4},{right_f:.4}]') 

        if abs(right_input - left_input) <= 1e-2:
            if verbose:
                print('Converged.')
            break

    return input

def sort_preds(pi,yi):
    index = (pi.mean(0).argmax(1) == yi)
    return pi[:,index,:], pi[:,~index,:]

def aucroc(id_scores,ood_scores):
    '''
    INPUTS: scores should be maximum softmax probability for each test example
    '''
    labels = np.zeros((id_scores.shape[0] + ood_scores.shape[0]), dtype=np.int32)
    labels[:id_scores.shape[0]] += 1
    examples = np.squeeze(np.hstack((id_scores, ood_scores)))
    return sk.roc_auc_score(labels, examples)

