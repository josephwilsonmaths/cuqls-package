import torch
from torch.func import functional_call
from torch.utils.data import DataLoader
import tqdm
import copy
import cuqls.utils as utils
from torchmetrics.classification import MulticlassCalibrationError

torch.set_default_dtype(torch.float64)

class classificationParallel(object):
    def __init__(self, network):
        self.network = network
        self.network.eval()
        self.device = next(network.parameters()).device
        print(f'CUQLS is using device {self.device}.')

        self.params = tuple(self.network.parameters())
        child_list = list(network.children())
        if len(child_list) > 1:
            child_list = child_list
        elif len(child_list) == 1:
            child_list = child_list[0]
        first_layers = child_list[:-1]
        final_layer = child_list[-1]
        self.llp = list(final_layer.parameters())[0].size().numel()
        self.num_output = list(final_layer.parameters())[0].shape[0]
        self.ck = torch.nn.Sequential(*first_layers)
        self.llp = list(self.ck.parameters())[-1].shape[0]
        self.theta_t = utils.flatten(self.params)[-(self.llp*self.num_output+self.num_output):-self.num_output]
        self.scale_cal = 1 # this will multiply the predictions, can change using ternary search to optimise ECE.
        self.J = None

    def jvp(self, x, v):
        if self.J is None:
            j = self.ck(x.to(self.device)).detach().reshape(x.shape[0], self.llp)
        else:
            j = self.J
        X = v.reshape(self.num_output, self.llp, -1).permute(1,0,2)
        return torch.einsum('np,pcs->ncs', j,X).detach().to(self.device)
    
    def vjp(self, x,V):
        # V : x.shape[0] x self.n_output x self.S
        if self.J is None:
            jt = self.ck(x.to(self.device)).detach().reshape(x.shape[0], self.llp).T
        else:
            jt = self.J.T
        return torch.einsum('pn,ncs->pcs',jt,V).detach().to(self.device)
    
    def lr_sched(self,lr,epoch,epochs,scheduler):
        if scheduler:
            m = (lr - lr/10) / epochs
            return lr/10 + m*(epochs - epoch)
        else:
            return lr
    
    def train(self, train, batchsize, S, scale, epochs, lr, mu, threshold = None, scheduler=False, verbose=False, extra_verbose=False, save_weights=None):
        train_loader = DataLoader(train,batchsize)

        if batchsize == len(train):
            X,_ = next(iter(train_loader))
            self.J = self.ck(X.to(self.device)).detach().reshape(X.shape[0], self.llp)
        else:
            self.J = None

        self.theta = torch.randn(size=(self.llp*self.num_output,S),device=self.device)*scale + self.theta_t.unsqueeze(1)
        
        if verbose:
            pbar = tqdm.trange(epochs)
        else:
            pbar = range(epochs)

        if save_weights is not None:
            save_dict = {}
            save_dict['info'] = {
                                'gamma': scale,
                                'epochs': epochs,
                                'lr': lr,
                                }
            save_dict['training'] = {}
        
        with torch.no_grad():
            for epoch in pbar:
                loss = torch.zeros((S), device='cpu')
                acc = torch.zeros((S), device='cpu')

                if extra_verbose:
                    pbar_inner = tqdm.tqdm(train_loader)
                else:
                    pbar_inner = train_loader

                for x,y in pbar_inner:
                    x, y = x.to(device=self.device, non_blocking=True, dtype=torch.float64), y.to(device=self.device, non_blocking=True)
                    f_nlin = self.network(x)
                    f_lin = (self.jvp(x, (self.theta.to(self.device) - self.theta_t.unsqueeze(1))).flatten(0,1) + 
                        f_nlin.reshape(-1,1)).reshape(x.shape[0],self.num_output,S)
                    if self.num_output > 1:
                        Mubar = torch.clamp(torch.nn.functional.softmax(f_lin,dim=1),1e-32,1)
                        ybar = torch.nn.functional.one_hot(y,num_classes=self.num_output)
                    else:
                        Mubar = torch.clamp(torch.nn.functional.sigmoid(f_lin),1e-32,1)
                        ybar = y.unsqueeze(1)
                    resid = (Mubar - ybar.unsqueeze(2))
                    g = self.vjp(x,resid).permute(1,0,2).flatten(0,1) / x.shape[0]

                    if epoch == 0:
                        bt = g
                    else:
                        bt = mu*bt + g

                    if self.num_output > 1:
                        l = (-1 / x.shape[0] * torch.sum((ybar.unsqueeze(2) * torch.log(Mubar)),dim=(0,1))).detach().cpu()
                    else:
                        input1 = torch.clamp(-f_lin,max = 0) + torch.log(1 + torch.exp(f_lin.abs()))
                        input2 = -f_lin - input1
                        l = - torch.sum((ybar.unsqueeze(2) * (-input1) + (1 - ybar).unsqueeze(2) * input2),dim=(0,1)) / x.shape[0]
                    loss += l

                    # Early stopping
                    if threshold:
                        bt[:,l <= threshold] = 0.0

                    self.theta -= self.lr_sched(lr,epoch,epochs,scheduler)*bt

                    if self.num_output > 1:
                        a = (f_lin.argmax(1) == y.unsqueeze(1).repeat(1,S)).type(torch.float).sum(0).detach().cpu()  # f_lin: N x C x S, y: N
                    else:
                        a = (f_lin.sigmoid().round().squeeze(1) == y.unsqueeze(1).repeat(1,S)).type(torch.float).sum(0).detach().cpu()  # f_lin: N x C x S, y: N
                    acc += a

                    if extra_verbose:
                        ma_l = loss / (pbar_inner.format_dict['n'] + 1)
                        ma_a = acc / ((pbar_inner.format_dict['n'] + 1) * x.shape[0])
                        metrics = {'min_loss_ma': ma_l.min().item(),
                                'max_loss_batch': ma_l.max().item(),
                                'min_acc_batch': ma_a.min().item(),
                                'max_acc_batch': ma_a.max().item(),
                                    'resid_norm': torch.mean(torch.square(g)).item()}
                        if self.device.type == 'cuda':
                            metrics['gpu_mem'] = 1e-9*torch.cuda.max_memory_allocated()
                        else:
                            metrics['gpu_mem'] = 0
                        if scheduler:
                            metrics['lr'] = self.lr_sched(lr,epoch,epochs,scheduler)
                        pbar_inner.set_postfix(metrics)

                loss /= len(train_loader)
                acc /= len(train)

                if verbose:
                    metrics = {'min_loss': loss.min().item(),
                               'max_loss': loss.max().item(),
                               'min_acc': acc.min().item(),
                               'max_acc': acc.max().item(),
                                'resid_norm': torch.mean(torch.square(g)).item()}
                    if self.device.type == 'cuda':
                        metrics['gpu_mem'] = 1e-9*torch.cuda.max_memory_allocated()
                    else:
                        metrics['gpu_mem'] = 0
                    if scheduler:
                        metrics['lr'] = self.lr_sched(lr,epoch,epochs,scheduler)
                    pbar.set_postfix(metrics)

                if save_weights is not None:
                    save_dict['training'][f'{epoch}'] = {
                                'weights': self.theta,
                                'min_loss': loss.min().item(),
                                'max_loss': loss.max().item(),
                                'min_acc': acc.min().item(),
                                'max_acc': acc.max().item(),
                                'resid_norm': torch.mean(torch.square(g)).item()
                                }
                    torch.save(save_dict, save_weights)

        if verbose:
            print('Posterior samples computed!')
        self.J = None

        return loss.max().item(), acc.min().item()

    def pre_load(self, pre_load):
        weight_dict = torch.load(pre_load, map_location=self.device)
        l = list(weight_dict['training'].keys())[-1]
        self.theta = weight_dict['training'][l]['weights']
        
    def test(self, test, test_bs=50, network_mean=False):

        test_loader = DataLoader(test, test_bs)
        
        # Concatenate predictions
        preds = []
        if network_mean:
            net_preds = []
        for x,_ in test_loader:
            print(f'loop mem: {1e-9*torch.cuda.max_memory_allocated()}')
            preds.append(self.eval(x).detach()) # S x N x C
            if network_mean:
                net_preds.append(self.network(x.to(self.device)).detach())
        predictions = torch.cat(preds,dim=1) # N x C x S ---> S x N x C
        if network_mean:
            predictions_net = torch.cat(net_preds,dim=0)    
            return predictions, predictions_net
        else:
            return predictions
    
    def UncertaintyPrediction(self, test, test_bs, network_mean=False):
        predictions = self.test(test, test_bs, network_mean)

        if not network_mean:
            probits = predictions.softmax(dim=-1)
            mean_prob = probits.mean(0)
            var_prob = probits.var(0)
        else:
            cuql_predictions, net_predictions = predictions
            mean_prob = net_predictions.softmax(dim=-1)
            var_prob = cuql_predictions.softmax(dim=-1).var(0)

        return mean_prob.detach().cpu(), var_prob.detach().cpu()
    
    def eval(self,x):
        x = x.to(self.device)
        f_nlin = self.network(x)
        f_lin = (self.jvp(x, (self.theta.to(self.device) - self.theta_t.unsqueeze(1))).flatten(0,1) + 
                            f_nlin.reshape(-1,1)).reshape(x.shape[0],self.num_output,-1)
        return f_lin.detach().permute(2,0,1).detach() * self.scale_cal
    
    def HyperparameterTuning(self, validation, metric = 'ece', left = 1e-2, right = 1e2, its = 100, verbose=False):
        predictions = self.test(validation, test_bs=200)

        loader = DataLoader(validation,batch_size=100)
        val_targets = torch.cat([y for _,y in loader])

        if metric == 'ece':
            ece_compute = MulticlassCalibrationError(num_classes=self.num_output,n_bins=10,norm='l1')
            def ece_eval(gamma):
                mean_prediction = (predictions * gamma).softmax(-1).mean(0)
                return ece_compute(mean_prediction, val_targets).cpu().item()
            f = ece_eval
            input_name, output_name = 'scale', 'ECE' 
        elif metric == 'varroc-id':
            def varroc_id_eval(gamma):
                scaled_predictions = (predictions * gamma).softmax(-1)
                idc, idic = utils.sort_preds(scaled_predictions,val_targets)
                return utils.aucroc(idic.var(0).sum(1), idc.var(0).sum(1))
            f = lambda x : -varroc_id_eval(x)
            input_name, output_name = 'scale', 'VARROC-ID' 
        else:
            print('Invalid metric choice. Valid choices are: [ece].')

        scale = utils.ternary_search(f = f,
                               left=left,
                               right=right,
                               its=its,
                               verbose=verbose,
                               input_name=input_name,
                               output_name=output_name)

        self.scale_cal = scale

class classificationParallelInterpolation(object):
    def __init__(self, network):
        self.network = network
        self.network.eval()
        self.device = next(network.parameters()).device
        print(f'CUQLS is using device {self.device}.')

        self.params = tuple(self.network.parameters())
        child_list = list(network.children())
        if len(child_list) > 1:
            child_list = child_list
        elif len(child_list) == 1:
            child_list = child_list[0]
        first_layers = child_list[:-1]
        final_layer = child_list[-1]
        self.llp = list(final_layer.parameters())[0].size().numel()
        self.num_output = list(final_layer.parameters())[0].shape[0]
        self.ck = torch.nn.Sequential(*first_layers)
        self.llp = list(self.ck.parameters())[-1].shape[0]
        self.theta_t = utils.flatten(self.params)[-(self.llp*self.num_output+self.num_output):-self.num_output]
        self.scale_cal = 1 # this will multiply the predictions, can change using ternary search to optimise ECE.
        self.J = None

    def jvp(self, x, v):
        if self.J is None:
            j = self.ck(x.to(self.device)).detach().reshape(x.shape[0], self.llp)
        else:
            j = self.J
        X = v.reshape(self.num_output, self.llp, -1).permute(1,0,2)
        return torch.einsum('np,pcs->ncs', j,X).detach().to(self.device)
    
    def vjp(self, x,V):
        # V : x.shape[0] x self.n_output x self.S
        if self.J is None:
            jt = self.ck(x.to(self.device)).detach().reshape(x.shape[0], self.llp).T
        else:
            jt = self.J.T
        return torch.einsum('pn,ncs->pcs',jt,V).detach().to(self.device)

    def train(self, train, batchsize, S, scale, epochs, lr, mu, decay = None, verbose=False, extra_verbose=False, save_weights=None):
        
        train_loader = DataLoader(train,batchsize)

        if batchsize == len(train):
            X,_ = next(iter(train_loader))
            self.J = self.ck(X.to(self.device)).detach().reshape(X.shape[0], self.llp)
        else:
            self.J = None

        self.theta = torch.randn(size=(self.llp*self.num_output,S),device=self.device)*scale + self.theta_t.unsqueeze(1)
        
        if verbose:
            pbar = tqdm.trange(epochs)
        else:
            pbar = range(epochs)

        if save_weights is not None:
            save_dict = {}
            save_dict['info'] = {
                                'gamma': scale,
                                'epochs': epochs,
                                'lr': lr,
                                }
            save_dict['training'] = {}
        
        with torch.no_grad():
            for epoch in pbar:
                loss = 0

                if extra_verbose:
                    pbar_inner = tqdm.tqdm(train_loader)
                else:
                    pbar_inner = train_loader

                for x,_ in pbar_inner:
                    x = x.to(device=self.device, non_blocking=True, dtype=torch.float64)
                    f = self.jvp(x, (self.theta.to(self.device) - self.theta_t.unsqueeze(1))) # N x C x S
                    g = self.vjp(x,f).permute(1,0,2).flatten(0,1) / x.shape[0] # P x C x S -> C x P x S -> CP x S

                    if epoch == 0:
                        bt = g
                    else:
                        bt = mu*bt + g

                    l = (torch.square(f).sum()  / (x.shape[0] * self.num_output * S)).detach()
                    loss += l
                    if decay is not None:
                        self.theta -= (lr * (decay ** epoch))*bt
                    else:
                        self.theta -= lr*bt

                    if extra_verbose:
                        metrics = {'loss': l.item(),
                                'resid_norm': torch.mean(torch.square(g)).item()}
                        if self.device.type == 'cuda':
                            metrics['gpu_mem'] = 1e-9*torch.cuda.max_memory_allocated()
                        else:
                            metrics['gpu_mem'] = 0
                        if decay is not None:
                            metrics['lr'] = (lr * (decay ** epoch))
                        pbar_inner.set_postfix(metrics)
                
                loss /= len(train_loader)

                if verbose:
                    metrics = {'loss': loss.item(),
                                'resid_norm': torch.mean(torch.square(g)).item()}
                    if self.device.type == 'cuda':
                        metrics['gpu_mem'] = 1e-9*torch.cuda.max_memory_allocated()
                    else:
                        metrics['gpu_mem'] = 0
                    if decay is not None:
                            metrics['lr'] = (lr * (decay ** epoch))
                    pbar.set_postfix(metrics)

                if save_weights is not None:
                    save_dict['training'][f'{epoch}'] = {
                                'weights': self.theta,
                                'loss': loss.item(),
                                'resid_norm': torch.mean(torch.square(g)).item()
                                }
                    torch.save(save_dict, save_weights)

        if verbose:
            print('Posterior samples computed!')
        self.J = None

        return loss.item()
    
    def pre_load(self, pre_load):
        weight_dict = torch.load(pre_load, map_location=self.device)
        l = list(weight_dict['training'].keys())[-1]
        self.theta = weight_dict['training'][l]['weights']
        
    import torch
from torch.func import functional_call
from torch.utils.data import DataLoader
import tqdm
import copy
import cuqls.utils as utils
from torchmetrics.classification import MulticlassCalibrationError

torch.set_default_dtype(torch.float64)

class classificationParallel(object):
    def __init__(self, network):
        self.network = network
        self.network.eval()
        self.device = next(network.parameters()).device
        print(f'CUQLS is using device {self.device}.')

        self.params = tuple(self.network.parameters())
        child_list = list(network.children())
        if len(child_list) > 1:
            child_list = child_list
        elif len(child_list) == 1:
            child_list = child_list[0]
        first_layers = child_list[:-1]
        final_layer = child_list[-1]
        self.llp = list(final_layer.parameters())[0].size().numel()
        self.num_output = list(final_layer.parameters())[0].shape[0]
        self.ck = torch.nn.Sequential(*first_layers)
        self.llp = list(self.ck.parameters())[-1].shape[0]
        self.theta_t = utils.flatten(self.params)[-(self.llp*self.num_output+self.num_output):-self.num_output]
        self.scale_cal = 1 # this will multiply the predictions, can change using ternary search to optimise ECE.
        self.J = None

    def jvp(self, x, v):
        if self.J is None:
            j = self.ck(x.to(self.device)).detach().reshape(x.shape[0], self.llp)
        else:
            j = self.J
        X = v.reshape(self.num_output, self.llp, -1).permute(1,0,2)
        return torch.einsum('np,pcs->ncs', j,X).detach().to(self.device)
    
    def vjp(self, x,V):
        # V : x.shape[0] x self.n_output x self.S
        if self.J is None:
            jt = self.ck(x.to(self.device)).detach().reshape(x.shape[0], self.llp).T
        else:
            jt = self.J.T
        return torch.einsum('pn,ncs->pcs',jt,V).detach().to(self.device)
    
    def lr_sched(self,lr,epoch,epochs,scheduler):
        if scheduler:
            m = (lr - lr/10) / epochs
            return lr/10 + m*(epochs - epoch)
        else:
            return lr
    
    def train(self, train, batchsize, S, scale, epochs, lr, mu, threshold = None, scheduler=False, verbose=False, extra_verbose=False, save_weights=None):
        train_loader = DataLoader(train,batchsize)

        if batchsize == len(train):
            X,_ = next(iter(train_loader))
            self.J = self.ck(X.to(self.device)).detach().reshape(X.shape[0], self.llp)
        else:
            self.J = None

        self.theta = torch.randn(size=(self.llp*self.num_output,S),device=self.device)*scale + self.theta_t.unsqueeze(1)
        
        if verbose:
            pbar = tqdm.trange(epochs)
        else:
            pbar = range(epochs)

        if save_weights is not None:
            save_dict = {}
            save_dict['info'] = {
                                'gamma': scale,
                                'epochs': epochs,
                                'lr': lr,
                                }
            save_dict['training'] = {}
        
        with torch.no_grad():
            for epoch in pbar:
                loss = torch.zeros((S), device='cpu')
                acc = torch.zeros((S), device='cpu')

                if extra_verbose:
                    pbar_inner = tqdm.tqdm(train_loader)
                else:
                    pbar_inner = train_loader

                for x,y in pbar_inner:
                    x, y = x.to(device=self.device, non_blocking=True, dtype=torch.float64), y.to(device=self.device, non_blocking=True)
                    f_nlin = self.network(x)
                    f_lin = (self.jvp(x, (self.theta.to(self.device) - self.theta_t.unsqueeze(1))).flatten(0,1) + 
                        f_nlin.reshape(-1,1)).reshape(x.shape[0],self.num_output,S)
                    if self.num_output > 1:
                        Mubar = torch.clamp(torch.nn.functional.softmax(f_lin,dim=1),1e-32,1)
                        ybar = torch.nn.functional.one_hot(y,num_classes=self.num_output)
                    else:
                        Mubar = torch.clamp(torch.nn.functional.sigmoid(f_lin),1e-32,1)
                        ybar = y.unsqueeze(1)
                    resid = (Mubar - ybar.unsqueeze(2))
                    g = self.vjp(x,resid).permute(1,0,2).flatten(0,1) / x.shape[0]

                    if epoch == 0:
                        bt = g
                    else:
                        bt = mu*bt + g

                    if self.num_output > 1:
                        l = (-1 / x.shape[0] * torch.sum((ybar.unsqueeze(2) * torch.log(Mubar)),dim=(0,1))).detach().cpu()
                    else:
                        input1 = torch.clamp(-f_lin,max = 0) + torch.log(1 + torch.exp(f_lin.abs()))
                        input2 = -f_lin - input1
                        l = - torch.sum((ybar.unsqueeze(2) * (-input1) + (1 - ybar).unsqueeze(2) * input2),dim=(0,1)) / x.shape[0]
                    loss += l

                    # Early stopping
                    if threshold:
                        bt[:,l <= threshold] = 0.0

                    self.theta -= self.lr_sched(lr,epoch,epochs,scheduler)*bt

                    if self.num_output > 1:
                        a = (f_lin.argmax(1) == y.unsqueeze(1).repeat(1,S)).type(torch.float).sum(0).detach().cpu()  # f_lin: N x C x S, y: N
                    else:
                        a = (f_lin.sigmoid().round().squeeze(1) == y.unsqueeze(1).repeat(1,S)).type(torch.float).sum(0).detach().cpu()  # f_lin: N x C x S, y: N
                    acc += a

                    if extra_verbose:
                        ma_l = loss / (pbar_inner.format_dict['n'] + 1)
                        ma_a = acc / ((pbar_inner.format_dict['n'] + 1) * x.shape[0])
                        metrics = {'min_loss_ma': ma_l.min().item(),
                                'max_loss_batch': ma_l.max().item(),
                                'min_acc_batch': ma_a.min().item(),
                                'max_acc_batch': ma_a.max().item(),
                                    'resid_norm': torch.mean(torch.square(g)).item()}
                        if self.device.type == 'cuda':
                            metrics['gpu_mem'] = 1e-9*torch.cuda.max_memory_allocated()
                        else:
                            metrics['gpu_mem'] = 0
                        if scheduler:
                            metrics['lr'] = self.lr_sched(lr,epoch,epochs,scheduler)
                        pbar_inner.set_postfix(metrics)

                loss /= len(train_loader)
                acc /= len(train)

                if verbose:
                    metrics = {'min_loss': loss.min().item(),
                               'max_loss': loss.max().item(),
                               'min_acc': acc.min().item(),
                               'max_acc': acc.max().item(),
                                'resid_norm': torch.mean(torch.square(g)).item()}
                    if self.device.type == 'cuda':
                        metrics['gpu_mem'] = 1e-9*torch.cuda.max_memory_allocated()
                    else:
                        metrics['gpu_mem'] = 0
                    if scheduler:
                        metrics['lr'] = self.lr_sched(lr,epoch,epochs,scheduler)
                    pbar.set_postfix(metrics)

                if save_weights is not None:
                    save_dict['training'][f'{epoch}'] = {
                                'weights': self.theta,
                                'min_loss': loss.min().item(),
                                'max_loss': loss.max().item(),
                                'min_acc': acc.min().item(),
                                'max_acc': acc.max().item(),
                                'resid_norm': torch.mean(torch.square(g)).item()
                                }
                    torch.save(save_dict, save_weights)

        if verbose:
            print('Posterior samples computed!')
        self.J = None

        return loss.max().item(), acc.min().item()

    def pre_load(self, pre_load):
        weight_dict = torch.load(pre_load, map_location=self.device)
        l = list(weight_dict['training'].keys())[-1]
        self.theta = weight_dict['training'][l]['weights']
        
    def test(self, test, test_bs=50, network_mean=False):

        test_loader = DataLoader(test, test_bs)
        
        # Concatenate predictions
        preds = []
        if network_mean:
            net_preds = []
        for x,_ in test_loader:
            preds.append(self.eval(x)) # S x N x C
            if network_mean:
                net_preds.append(self.network(x.to(self.device)).detach())
        predictions = torch.cat(preds,dim=1) # N x C x S ---> S x N x C
        if network_mean:
            predictions_net = torch.cat(net_preds,dim=0)    
            return predictions, predictions_net
        else:
            return predictions
    
    def UncertaintyPrediction(self, test, test_bs, network_mean=False):
        predictions = self.test(test, test_bs, network_mean)

        if not network_mean:
            probits = predictions.softmax(dim=-1)
            mean_prob = probits.mean(0)
            var_prob = probits.var(0)
        else:
            cuql_predictions, net_predictions = predictions
            mean_prob = net_predictions.softmax(dim=-1)
            var_prob = cuql_predictions.softmax(dim=-1).var(0)

        return mean_prob.detach().cpu(), var_prob.detach().cpu()
    
    def eval(self,x):
        x = x.to(self.device)
        f_nlin = self.network(x)
        f_lin = (self.jvp(x, (self.theta.to(self.device) - self.theta_t.unsqueeze(1))).flatten(0,1) + 
                            f_nlin.reshape(-1,1)).reshape(x.shape[0],self.num_output,-1)
        return f_lin.detach().permute(2,0,1) * self.scale_cal
    
    def HyperparameterTuning(self, validation, metric = 'ece', left = 1e-2, right = 1e2, its = 100, verbose=False):
        predictions = self.test(validation, test_bs=200)

        loader = DataLoader(validation,batch_size=100)
        val_targets = torch.cat([y for _,y in loader])

        if metric == 'ece':
            ece_compute = MulticlassCalibrationError(num_classes=self.num_output,n_bins=10,norm='l1')
            def ece_eval(gamma):
                mean_prediction = (predictions * gamma).softmax(-1).mean(0)
                return ece_compute(mean_prediction, val_targets).cpu().item()
            f = ece_eval
            input_name, output_name = 'scale', 'ECE' 
        elif metric == 'varroc-id':
            def varroc_id_eval(gamma):
                scaled_predictions = (predictions * gamma).softmax(-1)
                idc, idic = utils.sort_preds(scaled_predictions,val_targets)
                return utils.aucroc(idic.var(0).sum(1), idc.var(0).sum(1))
            f = lambda x : -varroc_id_eval(x)
            input_name, output_name = 'scale', 'VARROC-ID' 
        else:
            print('Invalid metric choice. Valid choices are: [ece].')

        scale = utils.ternary_search(f = f,
                               left=left,
                               right=right,
                               its=its,
                               verbose=verbose,
                               input_name=input_name,
                               output_name=output_name)

        self.scale_cal = scale

class classificationParallelInterpolation(object):
    def __init__(self, network):
        self.network = network
        self.network.eval()
        self.device = next(network.parameters()).device
        print(f'CUQLS is using device {self.device}.')

        self.params = tuple(self.network.parameters())
        child_list = list(network.children())
        if len(child_list) > 1:
            child_list = child_list
        elif len(child_list) == 1:
            child_list = child_list[0]
        first_layers = child_list[:-1]
        final_layer = child_list[-1]
        self.llp = list(final_layer.parameters())[0].size().numel()
        self.num_output = list(final_layer.parameters())[0].shape[0]
        self.ck = torch.nn.Sequential(*first_layers)
        self.llp = list(self.ck.parameters())[-1].shape[0]
        self.theta_t = utils.flatten(self.params)[-(self.llp*self.num_output+self.num_output):-self.num_output]
        self.scale_cal = 1 # this will multiply the predictions, can change using ternary search to optimise ECE.
        self.J = None

    def jvp(self, x, v):
        if self.J is None:
            j = self.ck(x.to(self.device)).detach().reshape(x.shape[0], self.llp)
        else:
            j = self.J
        X = v.reshape(self.num_output, self.llp, -1).permute(1,0,2)
        return torch.einsum('np,pcs->ncs', j,X).detach().to(self.device)
    
    def vjp(self, x,V):
        # V : x.shape[0] x self.n_output x self.S
        if self.J is None:
            jt = self.ck(x.to(self.device)).detach().reshape(x.shape[0], self.llp).T
        else:
            jt = self.J.T
        return torch.einsum('pn,ncs->pcs',jt,V).detach().to(self.device)

    def train(self, train, batchsize, S, scale, epochs, lr, mu, decay = None, verbose=False, extra_verbose=False, save_weights=None):
        
        train_loader = DataLoader(train,batchsize)

        if batchsize == len(train):
            X,_ = next(iter(train_loader))
            self.J = self.ck(X.to(self.device)).detach().reshape(X.shape[0], self.llp)
        else:
            self.J = None

        self.theta = torch.randn(size=(self.llp*self.num_output,S),device=self.device)*scale + self.theta_t.unsqueeze(1)
        
        if verbose:
            pbar = tqdm.trange(epochs)
        else:
            pbar = range(epochs)

        if save_weights is not None:
            save_dict = {}
            save_dict['info'] = {
                                'gamma': scale,
                                'epochs': epochs,
                                'lr': lr,
                                }
            save_dict['training'] = {}
        
        with torch.no_grad():
            for epoch in pbar:
                loss = 0
                ce_loss = torch.zeros((S), device='cpu')
                acc = torch.zeros((S), device='cpu')

                if extra_verbose:
                    pbar_inner = tqdm.tqdm(train_loader)
                else:
                    pbar_inner = train_loader

                for x,y in pbar_inner:
                    # Compute gradient
                    x, y = x.to(device=self.device, non_blocking=True, dtype=torch.float64), y.to(device=self.device, non_blocking=True)
                    f = self.jvp(x, (self.theta.to(self.device) - self.theta_t.unsqueeze(1))) # N x C x S
                    g = self.vjp(x,f).permute(1,0,2).flatten(0,1) / x.shape[0] # P x C x S -> C x P x S -> CP x S

                    # Momentum
                    if epoch == 0:
                        bt = g
                    else:
                        bt = mu*bt + g

                    # Compute squared-error loss
                    l = (torch.square(f).sum()  / (x.shape[0] * self.num_output * S)).detach()
                    loss += l

                    # Compute cross-entropy loss
                    f_nlin = self.network(x)

                    f_lin = (f.flatten(0,1) + f_nlin.reshape(-1,1)).reshape(x.shape[0],self.num_output,S)
                    if self.num_output > 1:
                        Mubar = torch.clamp(torch.nn.functional.softmax(f_lin,dim=1),1e-32,1)
                        ybar = torch.nn.functional.one_hot(y,num_classes=self.num_output)
                    else:
                        Mubar = torch.clamp(torch.nn.functional.sigmoid(f_lin),1e-32,1)
                        ybar = y.unsqueeze(1)

                    if self.num_output > 1:
                        ce_l = (-1 / x.shape[0] * torch.sum((ybar.unsqueeze(2) * torch.log(Mubar)),dim=(0,1))).detach().cpu()
                    else:
                        input1 = torch.clamp(-f_lin,max = 0) + torch.log(1 + torch.exp(f_lin.abs()))
                        input2 = -f_lin - input1
                        ce_l = - torch.sum((ybar.unsqueeze(2) * (-input1) + (1 - ybar).unsqueeze(2) * input2),dim=(0,1)) / x.shape[0]
                    ce_loss += ce_l

                    # Compute accuracy
                    if self.num_output > 1:
                        a = (f_lin.argmax(1) == y.unsqueeze(1).repeat(1,S)).type(torch.float).sum(0).detach().cpu()  # f_lin: N x C x S, y: N
                    else:
                        a = (f_lin.sigmoid().round().squeeze(1) == y.unsqueeze(1).repeat(1,S)).type(torch.float).sum(0).detach().cpu()  # f_lin: N x C x S, y: N
                    acc += a

                    # Iterate parameters
                    if decay is not None:
                        self.theta -= (lr * (decay ** epoch))*bt
                    else:
                        self.theta -= lr*bt

                    # Record metrics
                    if extra_verbose:
                        ma_l = ce_loss / (pbar_inner.format_dict['n'] + 1)
                        ma_a = acc / ((pbar_inner.format_dict['n'] + 1) * x.shape[0])
                        metrics = {'sq_loss': l.item(),
                                   'min_ce_loss': ma_l.min().item(),
                                    'max_ce_loss': ma_l.max().item(),
                                    'min_acc': ma_a.min().item(),
                                    'max_acc': ma_a.max().item(),
                                'resid_norm': torch.mean(torch.square(g)).item()}
                        if self.device.type == 'cuda':
                            metrics['gpu_mem'] = 1e-9*torch.cuda.max_memory_allocated()
                        else:
                            metrics['gpu_mem'] = 0
                        if decay is not None:
                            metrics['lr'] = (lr * (decay ** epoch))
                        pbar_inner.set_postfix(metrics)
                
                loss /= len(train_loader)
                ce_loss /= len(train_loader)
                acc /= len(train)

                if verbose:
                    metrics = {'sq_loss': loss.item(),
                                   'min_ce_loss': ce_loss.min().item(),
                                    'max_ce_loss': ce_loss.max().item(),
                                    'min_acc': acc.min().item(),
                                    'max_acc': acc.max().item(),
                                'resid_norm': torch.mean(torch.square(g)).item()}
                    if self.device.type == 'cuda':
                        metrics['gpu_mem'] = 1e-9*torch.cuda.max_memory_allocated()
                    else:
                        metrics['gpu_mem'] = 0
                    if decay is not None:
                            metrics['lr'] = (lr * (decay ** epoch))
                    pbar.set_postfix(metrics)

                if save_weights is not None:
                    save_dict['training'][f'{epoch}'] = {
                                'weights': self.theta,
                                'loss': loss.item(),
                                'resid_norm': torch.mean(torch.square(g)).item()
                                }
                    torch.save(save_dict, save_weights)

        if verbose:
            print('Posterior samples computed!')
        self.J = None

        return loss.item(), ce_loss.max().item(), acc.min().item()
    
    def pre_load(self, pre_load):
        weight_dict = torch.load(pre_load, map_location=self.device)
        l = list(weight_dict['training'].keys())[-1]
        self.theta = weight_dict['training'][l]['weights']
        
    def test(self, test, test_bs=50, network_mean=False):

        test_loader = DataLoader(test, test_bs)
        
        # Concatenate predictions
        preds = []
        if network_mean:
            net_preds = []
        for x,_ in test_loader:
            preds.append(self.eval(x)) # S x N x C
            if network_mean:
                net_preds.append(self.network(x.to(self.device)).detach())
        predictions = torch.cat(preds,dim=1) # N x C x S ---> S x N x C
        if network_mean:
            predictions_net = torch.cat(net_preds,dim=0)    
            return predictions, predictions_net
        else:
            return predictions
    
    def UncertaintyPrediction(self, test, test_bs, network_mean=False):
        predictions = self.test(test, test_bs, network_mean)

        if not network_mean:
            probits = predictions.softmax(dim=-1)
            mean_prob = probits.mean(0)
            var_prob = probits.var(0)
        else:
            cuql_predictions, net_predictions = predictions
            mean_prob = net_predictions.softmax(dim=-1)
            var_prob = cuql_predictions.softmax(dim=-1).var(0)

        return mean_prob.detach().cpu(), var_prob.detach().cpu()
    
    def eval(self,x):
        x = x.to(self.device)
        f_nlin = self.network(x)
        f_lin = (self.jvp(x, (self.theta.to(self.device) - self.theta_t.unsqueeze(1))).flatten(0,1) + 
                            f_nlin.reshape(-1,1)).reshape(x.shape[0],self.num_output,-1)
        return f_lin.detach().permute(2,0,1).detach() * self.scale_cal
    
    def HyperparameterTuning(self, validation, metric = 'ece', left = 1e-2, right = 1e2, its = 100, verbose=False):
        predictions = self.test(validation, test_bs=200)

        loader = DataLoader(validation,batch_size=100)
        val_targets = torch.cat([y for _,y in loader])

        if metric == 'ece':
            ece_compute = MulticlassCalibrationError(num_classes=self.num_output,n_bins=10,norm='l1')
            def ece_eval(gamma):
                mean_prediction = (predictions * gamma).softmax(-1).mean(0)
                return ece_compute(mean_prediction, val_targets).cpu().item()
            f = ece_eval
            input_name, output_name = 'scale', 'ECE' 
        elif metric == 'varroc-id':
            def varroc_id_eval(gamma):
                scaled_predictions = (predictions * gamma).softmax(-1)
                idc, idic = utils.sort_preds(scaled_predictions,val_targets)
                return utils.aucroc(idic.var(0).sum(1), idc.var(0).sum(1))
            f = lambda x : -varroc_id_eval(x)
            input_name, output_name = 'scale', 'VARROC-ID' 
        else:
            print('Invalid metric choice. Valid choices are: [ece].')

        scale = utils.ternary_search(f = f,
                               left=left,
                               right=right,
                               its=its,
                               verbose=verbose,
                               input_name=input_name,
                               output_name=output_name)

        self.scale_cal = scale