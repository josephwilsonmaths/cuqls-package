from abc import abstractmethod
import torch
from torch.utils.data import DataLoader
import tqdm
import matplotlib.pyplot as plt

class LinearSamplingPosterior:
    def __init__(self, network, precision='double'):
        self.network = network
        self.network.eval()
        self.device = next(network.parameters()).device
        self.dtype = (torch.float64 if precision == 'double' else (torch.float32 if precision == 'single' else torch.float16))
        self.scale_cal = 1.0

    def set_methodname(self, methodname):
        self.methodname = methodname

    @abstractmethod
    def jvp(self, x, v):
        pass

    @abstractmethod
    def vjp(self, x, V):
        pass

    @abstractmethod
    def compute_full_jacobian(self, x):
        pass

    @abstractmethod
    def sample_theta(self, S, gamma):
        pass    

    @abstractmethod
    def instantiate_lossdict(self):
        pass

    @abstractmethod
    def compute_gradient(self, x, y, theta):
        '''
        Note, if self.lossfunction is not least_squares_logits, J (theta - theta*) is returned as None.

        Takes x: input data, N x D, y: input labels, N, theta: parameters, P x S
        Returns f_lin: N x C x S, g: P x S, J (theta - theta*): N x C x S.
        '''
        pass

    @abstractmethod
    def compute_loss(self, f, resid, y, g):
        pass

    @abstractmethod
    def compute_flin(self, x):
        pass

    def compute_squared_error(self, residual):
        ''' Compute squared error ||J(theta - theta_star)||^2 = ||f_lin - f_nlin||^2 for classification 
        residual: N x C x S
        '''
        N,C,S = residual.shape
        l = (torch.square(residual).sum()  / (N * C * S)).detach()
        return l.mean().item()

    def compute_accuracy(self, f_lin, y):
        ''' Compute accuracy for classification tasks 
        f_lin: N x C x S
        y: N
        '''
        _,C,S = f_lin.shape
        # Compute accuracy
        if C > 1:
            a = (f_lin.argmax(1) == y.unsqueeze(1).repeat(1,S)).type(self.dtype).sum(0).detach().cpu()  # f_lin: N x C x S, y: N
        else:
            a = (f_lin.sigmoid().round().squeeze(1) == y.unsqueeze(1).repeat(1,S)).type(self.dtype).sum(0).detach().cpu()  # f_lin: N x C x S, y: N
        return (a / y.shape[0]).mean().item()
    
    def compute_cross_entropy(self, f_lin, y):
        ''' Compute cross-entropy for classification tasks 
        f_lin: N x C x S
        y: N
        '''
        N,C,S = f_lin.shape
        if C > 1:
            Mubar = torch.clamp(torch.nn.functional.softmax(f_lin,dim=1),torch.finfo(self.dtype).eps,1)
            ybar = torch.nn.functional.one_hot(y,num_classes=C)
        else:
            Mubar = torch.clamp(torch.nn.functional.sigmoid(f_lin),torch.finfo(self.dtype).eps,1)
            ybar = y.unsqueeze(1)

        if C > 1:
            ce = -torch.sum((ybar.unsqueeze(2) * torch.log(Mubar)),dim=(0,1)) / N
        else:
            input1 = torch.clamp(-f_lin,max = 0) + torch.log(1 + torch.exp(f_lin.abs()))
            input2 = -f_lin - input1
            ce = - torch.sum((ybar.unsqueeze(2) * (-input1) + (1 - ybar).unsqueeze(2) * input2),dim=(0,1)) / N
        return ce.mean(0).item()

    def metric_reporting(self, loss_dict, average='running'):
        report_dict = {}

        if average == 'running':
            an = 0
        elif average == 'moving':
            an = -100
        else:
            raise ValueError("average must be 'running' or 'moving'")
        
        for metric in loss_dict.keys():
            metric_value = torch.mean(torch.tensor(loss_dict[metric][-1][an:]))
            report_dict[f'mean_{metric}'] = metric_value.item()

        return report_dict
    
    def plot_loss_metrics(self, loss_dict, plot_loss_dir):
        metrics = loss_dict.keys()
        n_metrics = len(metrics)
        fig, axes = plt.subplots(1, n_metrics, figsize=(12, 4))

        if n_metrics == 1:
            axes = [axes]

        colors = plt.cm.tab10(range(n_metrics))

        for idx, (metric, ax, color) in enumerate(zip(metrics, axes, colors)):
            concatenated_values = torch.tensor(loss_dict[metric]).flatten().cpu().numpy()
            ax.plot(concatenated_values, color=color, linewidth=2)
            ax.set_title(metric)
            ax.set_xlabel('Iteration')
            ax.set_ylabel('Value')
            ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(plot_loss_dir + f"{self.methodname}_loss_plot.pdf", format='pdf',bbox_inches='tight')

    def train(self, 
              train, 
              bs, 
              n_output = 10, 
              gamma = 1, 
              S = 10, 
              epochs = 100, 
              lr = 1e-2, 
              mu = 0.9, 
              threshold = None, 
              verbose = False, 
              extra_verbose = False, 
              save_weights = None,
              plot_loss_dir = None,
              average = 'running'):
        
        '''
        Train linear sampling posterior using SGD with momentum
        train: training dataset
        bs: batch size
        n_output: number of output classes
        gamma: prior variance
        S: number of linear functions
        epochs: number of training epochs
        lr: learning rate
        mu: momentum factor for SGD
        threshold: early stopping threshold
        verbose: print progress bar
        extra_verbose: print inner progress bar
        save_weights: path to save weights
        plot_loss_dir: directory to save loss plots
        average: averaging method for metrics ('running' or 'moving')
        '''
        
        # Create data loader
        train_loader = DataLoader(train,bs)    

        # Creat full-batch Jacobian if batch size equals training set size
        if bs == len(train):
            X,_ = next(iter(train_loader))
            self.J = self.compute_full_jacobian(X.to(device=self.device, dtype=self.dtype))
        else:
            self.J = None

        # Instantiate theta
        self.theta = self.sample_theta(S, gamma)

        # Progress bar
        if verbose:
            pbar = tqdm.trange(epochs)
        else:
            pbar = range(epochs)

        # Save training history
        if save_weights is not None:
            save_dict = {}
            save_dict['info'] = {
                                'gamma': gamma,
                                'epochs': epochs,
                                'lr': lr,
                                }
            save_dict['training'] = {}
        
        loss_dict = self.instantiate_lossdict()

        for epoch in pbar:

            # Create epoch loss dict
            for metric in loss_dict.keys():
                loss_dict[metric].append([])

            # Per Epoch Progress bar
            if extra_verbose:
                pbar_inner = tqdm.tqdm(train_loader)
            else:
                pbar_inner = train_loader

            for x,y in pbar_inner:
                x, y = x.to(device=self.device, non_blocking=True, dtype=self.dtype), y.to(device=self.device, non_blocking=True)

                # Compute gradient
                flin, g, resid = self.compute_gradient(x, y, self.theta)

                # Momentum
                if epoch == 0:
                    bt = g
                else:
                    bt = mu*bt + g

                # Iterate parameters
                self.theta -= lr*bt

                # Compute loss
                loss_vals = self.compute_loss(flin,resid,y,g)
                for metric in loss_dict.keys():
                    loss_dict[metric][-1].append(loss_vals[metric])

                # Record metrics
                if extra_verbose:
                    metrics = self.metric_reporting(loss_dict, average=average)
                    if self.device.type == 'cuda':
                        metrics['gpu_mem'] = 1e-9*torch.cuda.max_memory_allocated()
                    else:
                        metrics['gpu_mem'] = 0
                    pbar_inner.set_postfix(metrics)
            
            # Record epoch metrics
            if verbose:
                metrics = self.metric_reporting(loss_dict, average='running')
                if self.device.type == 'cuda':
                    metrics['gpu_mem'] = 1e-9*torch.cuda.max_memory_allocated()
                else:
                    metrics['gpu_mem'] = 0
                pbar.set_postfix(metrics)

            if save_weights is not None:
                model_dict = {
                            'weights': self.theta,
                            }
                metrics = self.metric_reporting(loss_dict, average='running')
                model_dict.update(metrics)
                
                save_dict['training'][f'{epoch}'] = model_dict
                torch.save(save_dict, save_weights)

        if verbose:
            print('Linear sampling training complete.')
        self.J = None

        if plot_loss_dir is not None:
            self.plot_loss_metrics(loss_dict, plot_loss_dir)

        return self.metric_reporting(loss_dict, average='running')
    
    def pre_load(self, pre_load):
        ''' 
        Pre-load weights from a saved training session
        pre_load: path to saved weights
        '''
        weight_dict = torch.load(pre_load, map_location=self.device)
        l = list(weight_dict['training'].keys())[-1]
        self.theta = weight_dict['training'][l]['weights']

    def eval(self,x):
        ''' 
        Evaluate linear model at inputs x
        x: N x D
        returns: S x N x C
        '''
        x = x.to(device=self.device, dtype=self.dtype)
        f_lin = self.compute_flin(x) # N x C x S, should be detached from computation graph
        return f_lin.permute(2,0,1) * self.scale_cal # S x N x C
    
    def test(self, test, bs=50, network_mean=False, verbose=False):
        '''
        Compute predictions on test dataset for Linear Sampling posterior.
        test: test dataset
        bs: batch size for testing
        network_mean: whether to use network for mean predictions
        verbose: print progress bar
        '''

        # Create data loader
        test_loader = DataLoader(test, bs)
        
        # Concatenate predictions
        preds = []
        if network_mean:
            net_preds = []

        # Progress bar
        if verbose:
            pbar = tqdm.tqdm(test_loader)
        else:
            pbar = test_loader

        # Loop over batches
        for x,_ in pbar:

            # Compute predictions
            preds.append(self.eval(x)) # S x N x C

            # Compute network mean predictions
            if network_mean:
                net_preds.append(self.network(x.to(device=self.device, dtype=self.dtype)).detach()) # N x C

        # Concatenate predictions
        predictions = torch.cat(preds,dim=1) # S x N x C

        # Return predictions
        if network_mean:
            predictions_net = torch.cat(net_preds,dim=0)    
            return predictions, predictions_net
        else:
            return predictions
        
    def UncertaintyPrediction(self, test, bs, network_mean=False, verbose=False):
        '''
        Compute mean and variance of predictions on test dataset for Linear Sampling posterior.
        test: test dataset
        bs: batch size for testing
        network_mean: whether to use network for mean predictions
        verbose: print progress bar
        '''

        predictions = self.test(test, bs, network_mean, verbose) # Should be detached from computation graph, S x N x C or (S x N x C, N x C)

        if not network_mean:
            probits = predictions.softmax(dim=-1)
            mean_prob = probits.mean(0)
            var_prob = probits.var(0)
        else:
            linear_predictions, net_predictions = predictions
            mean_prob = net_predictions.softmax(dim=-1)
            var_prob = linear_predictions.softmax(dim=-1).var(0)

        return mean_prob.detach().cpu(), var_prob.detach().cpu()
    
    def HyperparameterTuning(self):
        raise NotImplementedError("Hyperparameter tuning not implemented for Linear Sampling Posterior.")