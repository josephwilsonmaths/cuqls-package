from LinearSampling.LinearSamplingPosterior import LinearSamplingPosterior
from LinearSampling.KernelFunction import NeuralTangentKernelSampler, ConjugateKernelSampler
import torch

class CrossEntropyPosterior(LinearSamplingPosterior):
    def __init__(self, kernel, network, precision='single'):
        super().__init__(network, precision)
        self.kernel = kernel

        if self.kernel == 'ntk':
            self.kernel_function = NeuralTangentKernelSampler(self.network, self.dtype)
        elif self.kernel == 'ck':
            self.kernel_function = ConjugateKernelSampler(self.network, self.dtype)
        else:
            raise ValueError('Invalid kernel type. Options are "ntk" or "ck".')

        self.lossfunction = 'cross_entropy'

    def sample_theta(self, S, gamma):
        return self.kernel_function.sample_theta(S, gamma)

    def instantiate_lossdict(self):
        loss_dict = {'ce loss': [],
                    'accuracy': [],
                    'grad norm': []}
        return loss_dict
    
    def compute_loss(self, f, resid, y, g):
        ce_loss = self.compute_cross_entropy(f, y)
        accuracy = self.compute_accuracy(f, y)
        grad_norm = torch.norm(g).item()
        loss_vals = {'ce loss': ce_loss,
                     'accuracy': accuracy,
                     'grad norm': grad_norm}
        return loss_vals
    
    def compute_gradient(self, x, y, theta):
        '''
        Compute gradient of cross-entropy loss for linear models
        x: input data, N x D
        y: input labels, N
        '''
        # Forward pass of linear models
        f_lin = self.kernel_function.compute_flin(x, self.theta, self.J) # N x C x S, should be detached from computation graph
        N, C, _ = f_lin.shape
        if C > 1:
            Mubar = torch.clamp(torch.nn.functional.softmax(f_lin,dim=1),1e-32,1)
            ybar = torch.nn.functional.one_hot(y,num_classes=C)
        else:
            Mubar = torch.clamp(torch.nn.functional.sigmoid(f_lin),1e-32,1)
            ybar = y.unsqueeze(1)
        resid = (Mubar - ybar.unsqueeze(2))
        g = self.kernel_function.vjp(x, resid, self.J).detach() / N # P x S, should be detached from computation graph
        return f_lin, g, None
    
    def compute_flin(self, x):
        return self.kernel_function.compute_flin(x, self.theta, self.J)


class LeastSquaresLogitsPosterior(LinearSamplingPosterior):
    def __init__(self, kernel, network, precision='single'):
        super().__init__(network, precision)
        self.kernel = kernel

        if self.kernel == 'ntk':
            self.kernel_function = NeuralTangentKernelSampler(self.network, self.dtype)
        elif self.kernel == 'ck':
            self.kernel_function = ConjugateKernelSampler(self.network, self.dtype)
        else:
            raise ValueError('Invalid kernel type. Options are "ntk" or "ck".')
        
        self.lossfunction = 'least_squares_logits'

    def sample_theta(self, S, gamma):
        return self.kernel_function.sample_theta(S, gamma)

    def instantiate_lossdict(self):
        loss_dict = {'sq loss': [],
                    'ce loss': [],
                    'accuracy': [],
                    'grad norm': []}
        return loss_dict
    
    def compute_loss(self, f, resid, y, g):
        sq_loss = self.compute_squared_error(resid)
        ce_loss = self.compute_cross_entropy(f, y)
        accuracy = self.compute_accuracy(f, y)
        grad_norm = torch.norm(g).item()
        loss_vals = {'sq loss': sq_loss,
                     'ce loss': ce_loss,
                     'accuracy': accuracy,
                     'grad norm': grad_norm}
        return loss_vals

    def compute_gradient(self, x, y, theta):
        f_nlin, proj = self.kernel_function.jvp(x, theta, self.J) # N x C x S, should be detached from computation graph
        f_lin = proj + f_nlin.unsqueeze(2) # N x C x S, should be detached from computation graph 
        N, C, _ = proj.shape
        g = self.kernel_function.vjp(x, proj, self.J).detach() / (N*C) # P x S, should be detached from computation graph
        return f_lin, g, proj
    
    def compute_flin(self, x):
        return self.kernel_function.compute_flin(x, self.theta, self.J)