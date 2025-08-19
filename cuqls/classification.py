import torch
from torch.func import functional_call
from torch.utils.data import DataLoader
import tqdm
import copy
import nuqls.utils as utils

torch.set_default_dtype(torch.float64)

class classificationParallel(object):
    def __init__(self, network):
        self.network = network
        self.network.eval()
        self.device = next(network.parameters()).device
        print(f'NUQLS is using device {self.device}.')

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

    def jvp(self, x, v):
        j = self.ck(x.to(self.device)).detach().reshape(x.shape[0], self.llp)
        X = v.reshape(self.num_output, self.llp, -1).permute(1,0,2)
        return torch.einsum('np,pcs->ncs', j,X).detach().to(self.device)
    
    def vjp(self, x,V):
        # V : x.shape[0] x self.n_output x self.S
        jt = self.ck(x.to(self.device)).detach().reshape(x.shape[0], self.llp).T
        return torch.einsum('pn,ncs->pcs',jt,V).detach().to(self.device)
    
    def lr_sched(self,lr,epoch,epochs,scheduler):
        if scheduler:
            m = (lr - lr/10) / epochs
            return lr/10 + m*(epochs - epoch)
        else:
            return lr
    
    def train(self, train, batchsize, S, scale, epochs, lr, mu, scheduler=False, verbose=False, extra_verbose=False, save_weights=None):
        train_loader = DataLoader(train,batchsize)

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
                    Mubar = torch.clamp(torch.nn.functional.softmax(f_lin,dim=1),1e-32,1)
                    ybar = torch.nn.functional.one_hot(y,num_classes=self.num_output)
                    g = self.vjp(x,(Mubar - ybar.unsqueeze(2))).permute(1,0,2).flatten(0,1) / x.shape[0]

                    if epoch == 0:
                        bt = g
                    else:
                        bt = mu*bt + g

                    self.theta -= self.lr_sched(lr,epoch,epochs,scheduler)*bt

                    l = (- 1 / x.shape[0] * torch.sum((ybar.flatten(0,1).unsqueeze(1) * torch.log(Mubar.flatten(0,1))),dim=0)).cpu()
                    loss += l

                    a = (f_lin.argmax(1) == y.unsqueeze(1).repeat(1,S)).type(torch.float).sum(0).detach().cpu()  # f_lin: N x C x S, y: N
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

        return loss.max().item(), acc.min().item()

    def pre_load(self, pre_load):
        weight_dict = torch.load(pre_load, map_location=self.device)
        l = list(weight_dict['training'].keys())[-1]
        self.theta_S = weight_dict['training'][l]['weights']
        
    def test(self, test, test_bs=50):

        test_loader = DataLoader(test, test_bs)
        
        # Concatenate predictions
        pred_s = []
        for x,y in test_loader:
            x, y = x.to(self.device), y.to(self.device)
            f_nlin = self.network(x)
            f_lin = (self.jvp(x, (self.theta.to(self.device) - self.theta_t.unsqueeze(1))).flatten(0,1) + 
                            f_nlin.reshape(-1,1)).reshape(x.shape[0],self.num_output,-1)
            pred_s.append(f_lin.detach())

        id_predictions = torch.cat(pred_s,dim=0).permute(2,0,1) # N x C x S ---> S x N x C
        
        del f_lin
        del pred_s
    
        return id_predictions
    
    def UncertaintyPrediction(self, test, test_bs):

        logits = self.test(test, test_bs)

        probits = logits.softmax(dim=2)
        mean_prob = probits.mean(0)
        var_prob = probits.var(0)

        return mean_prob.detach(), var_prob.detach()
    
    def eval(self,x):
        x = x.to(self.device)
        f_nlin = self.network(x)
        f_lin = (self.jvp(x, (self.theta.to(self.device) - self.theta_t.unsqueeze(1))).flatten(0,1) + 
                            f_nlin.reshape(-1,1)).reshape(x.shape[0],self.num_output,-1)
        return f_lin.detach().permute(2,0,1) 
    