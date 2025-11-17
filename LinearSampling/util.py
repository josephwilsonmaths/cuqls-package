import torch
import scipy
import numpy as np
import sklearn.metrics as sk
import tqdm


def _dub(x,y):
    return {yi:xi - y[yi] for (xi, yi) in zip(x, y)}

def flatten(lst):
    tmp = [i.contiguous().view(-1, 1) for i in lst]
    return torch.cat(tmp).view(-1)

def unflatten_like(vector, likeTensorList):
    # Takes a flat torch.tensor and unflattens it to a list of torch.tensors
    #    shaped like likeTensorList
    outList = []
    i = 0
    for tensor in likeTensorList:
        # n = module._parameters[name].numel()
        n = tensor.numel()
        outList.append(vector[i : i + n].view(tensor.shape))
        i += n
    return tuple(outList)


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

def varroc(testing_function, test_dataloader, ood_test_dataloader, verbose=False):

    if verbose:
        pbar = tqdm.tqdm(test_dataloader, desc='VARROC-ID Testing')
    else:
        pbar = test_dataloader

    idc = []
    idic = []

    for x, y in pbar:
        samplex = testing_function(x)
        mean, var = samplex.mean(dim=0), samplex.var(dim=0)

        # Find the index of predictions that are correct
        index_correct = sort_preds_index(mean,y)

        # Separate variance for correct and incorrect predictions
        correct_var = var[index_correct,:]
        incorrect_var = var[~index_correct,:]

        # For correct and incorrect predictions, find the index of the max predicted class
        max_index_correct = mean[index_correct,:].argmax(1)
        max_index_incorrect = mean[~index_correct,:].argmax(1)

        # Append the variance corresponding to the max predicted class, for correct and incorrect predictions
        idc.append(correct_var[range(len(max_index_correct)),max_index_correct])
        idic.append(incorrect_var[range(len(max_index_incorrect)),max_index_incorrect])

    varroc_id = aucroc(torch.cat(idic), torch.cat(idc))

    if verbose:
        pbar = tqdm.tqdm(ood_test_dataloader, desc='VARROC-OOD Testing')
    else:
        pbar = ood_test_dataloader

    ood = []

    for x, y in pbar:
        samplex = testing_function(x)
        mean, var = samplex.mean(dim=0), samplex.var(dim=0)

        # Find the index of the max predicted class
        max_index_correct = mean.argmax(1)

        # Append the variance corresponding to the max predicted class, for correct and incorrect predictions
        ood.append(var[range(len(max_index_correct)),max_index_correct])

    varroc_ood = aucroc(torch.cat(ood), torch.cat(idc))

    return varroc_id, varroc_ood


def sort_preds_index(pi,yi):
    return (pi.argmax(1) == yi)

