import torch
from functorch import make_functional
from torch.utils.data import DataLoader
import tqdm
import copy
import nuqls.utils as utils

torch.set_default_dtype(torch.float64)


class regressionParallel(object):
    '''
    NUQLS implementation for: regression, larger S, tiny dataset size. Per epoch training is much slower compared to serial implement, but overall is faster. Use for S > 10.
    '''
    def __init__(self,network):
        self.network = network
        self.device = next(network.parameters()).device

        child_list = list(self.network.children())
        if len(child_list) > 1:
            first_layers = child_list[:-1]
        elif len(child_list) == 1:
            first_layers = child_list[0][:-1]
        self.ck = torch.nn.Sequential(*first_layers)

        self.llp = list(self.ck.parameters())[-1].shape[0]

    def train(self, train, batchsize, scale=1, S=10, epochs=100, lr=1e-3, mu=0.9, verbose=False):
        _, params = make_functional(self.network)
        theta_t = utils.flatten(params)[-(self.llp+1):-1]
        theta = torch.randn(size=(self.llp,S),device=self.device)*scale + theta_t.unsqueeze(1)

        ## Create loaders and get entire training set
        train_loader = DataLoader(train,batch_size=batchsize)
    
        if batchsize == len(train):
            X,_ = next(iter(train_loader))
            J = self.ck(X.to(self.device)).detach()
        else:
            J = None

        def jacobian(x):
            if J is not None:
                return J
            else:
                return self.ck(x.to(self.device)).detach()

        # Set progress bar
        if verbose:
            pbar = tqdm.trange(epochs)
        else:
            pbar = range(epochs)
        
        ## Train S realisations of linearised networks
        for epoch in pbar:
            loss = 0
            resid_norm = 0

            if self.device == torch.device('cuda'):
                torch.cuda.reset_peak_memory_stats()

            for x,y in train_loader:
                x, y = x.to(self.device), y.to(self.device).reshape(-1,1)
                f_nlin = self.network(x)
                jacobian_batch = jacobian(x)
                f_lin = (jacobian_batch.to(self.device) @ (theta - theta_t.unsqueeze(1)) + f_nlin).detach()
                resid = f_lin - y
                g = jacobian_batch.T.to(self.device) @ resid.to(self.device) / x.shape[0]

                if epoch == 0:
                    bt = g
                else:
                    bt = mu*bt + g

                theta -= lr * bt

                loss += torch.mean(torch.square(jacobian_batch @ (theta - theta_t.unsqueeze(1)) + f_nlin - y)).max().item()
                resid_norm += torch.mean(torch.square(g)).item()

            loss /= len(train_loader)
            resid_norm /= len(train_loader)

            if verbose:
                metrics = {'max_loss': loss,
                   'resid_norm': resid_norm}
                if self.device == torch.device('cuda'):
                    metrics['gpu_mem'] = 1e-9*torch.cuda.max_memory_allocated()
                else:
                    metrics['gpu_mem'] = 0
                pbar.set_postfix(metrics)

        if verbose:
            print('Posterior samples computed!')

        self.theta = theta

        return loss, resid_norm
    
    def test(self, test, test_bs=1000):
        _, params = make_functional(self.network)
        theta_t = utils.flatten(params)[-(self.llp+1):-1]

        test_loader = DataLoader(test,batch_size=test_bs)
                
        # Concatenate predictions
        pred_s = []
        for x,y in test_loader:
            x, y = x.to(self.device), y.to(self.device)
            f_nlin = self.network(x)
            J = self.ck(x.to(self.device)).detach()

            pred_lin = J @ (self.theta - theta_t.unsqueeze(1)) + f_nlin ## n x S
            pred_s.append(pred_lin.detach())

        predictions = torch.cat(pred_s,dim=0)
        return predictions
    
    def HyperparameterTuning(self, validation, left, right, its, verbose=False):
        val_predictions = self.test(validation)

        calibration_test_loader_val = DataLoader(validation,len(validation))
        _, val_y = next(iter(calibration_test_loader_val))

        def ece_eval(gamma):
            scaled_nuqls_predictions = val_predictions * gamma
            obs_map, predicted = utils.calibration_curve_r(val_y,val_predictions.mean(1),scaled_nuqls_predictions.var(1),11)
            return torch.mean(torch.square(obs_map - predicted)).item()

        scale = utils.ternary_search(f = ece_eval,
                               left=left,
                               right=right,
                               its=its,
                               verbose=verbose,
                               input_name="scale",
                               output_name="ECE")

        self.scale_cal = scale

    def eval(self,x):
        _, params = make_functional(self.network)
        theta_t = utils.flatten(params)[-(self.llp+1):-1]

        x = x.to(self.device)
        f_nlin = self.network(x)
        J = self.ck(x.to(self.device)).detach()

        pred_lin = J @ (self.theta - theta_t.unsqueeze(1)) + f_nlin ## n x S
        return pred_lin.detach()