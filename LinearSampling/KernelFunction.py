import torch
from torch.func import functional_call, vmap, jacrev
import LinearSampling.util as util
import copy
    

class NeuralTangentKernelSampler(object):
    def __init__(self, network, dtype):
        self.network = network
        self.device = next(network.parameters()).device
        self.dtype = dtype
        self.params = {k: v.detach() for k, v in self.network.named_parameters()} # dict of parameters

    def fnet(self, params, x):
        return functional_call(self.network, params, x)
    
    def sample_theta(self, S, scale):
        theta = []
        p = copy.deepcopy(self.params).values()
        for s in range(S):
            theta_star = []
            for pi in p:
                theta_star.append(torch.nn.parameter.Parameter(torch.randn(size=pi.shape,device=self.device,dtype=self.dtype)*scale + pi.to(self.device)))
            theta.append(util.flatten(theta_star))
        
        theta = torch.stack(theta, dim=0).T # P x S
        return theta    

    def jvp_single(self,theta,params,x):
        dparams = util._dub(util.unflatten_like(theta, self.params.values()), self.params)
        f, proj = torch.func.jvp(lambda param: self.fnet(param, x),
                                (params,), (dparams,))
        f, proj = f.detach(), proj.detach()
        return f, proj
    
    def vjp_single(self,v,params,x):
        _, vjp_fn = torch.func.vjp(lambda param: self.fnet(param, x), params)
        vjp = vjp_fn(v)
        return vjp
    
    def compute_full_jacobian(self, x):
        J = vmap(jacrev(self.fnet), (None, 0))(self.params, x)
        J = [j.detach().flatten(2) for j in J.values()]
        J = torch.cat(J,dim=2).detach()
        F = self.fnet(self.params, x).detach()
        return (J, F)

    def jvp(self, x, theta, J):
        '''
        x: input data, N x D
        theta: parameters, P x S
        J = None or (Jac, F) where Jacobian is N x C x P and F is N x C, with N = len(training data), C = number of classes, P = number of parameters. 
        Note: J is expected to be detached from computation graph.
        '''

        # If we have access to full Jacobian J, use it
        if J is not None:
            Jac, F = J # N x C x P, N x C
            proj = Jac @ (theta - self.theta_t.unsqueeze(1)) # N x C x S
            return F, proj # N x C, N x C x S
        
        # Else, compute JVP via autograd
        else:
            # f_nlin, proj = torch.vmap(self.jvp_single, (1,None,None))((theta),self.params,x).permute(1,2,0) # N x C, N x C x S
            f_nlin, proj = torch.vmap(self.jvp_single, (1,None,None))((theta),self.params,x)
            return f_nlin.detach().permute(1,2,0)[:,:,0], proj.permute(1,2,0).detach()
        
    def compute_flin(self, x, theta, J):
        f_nlin, proj = self.jvp(x, theta, J) # N x C, N x C x S
        return f_nlin.unsqueeze(2) + proj # N x C x S
    
    def vjp(self, x, v, J):
        '''
        x: input data, N x D
        v: N x C x S
        Returns: P x S
        '''
        if J is not None:
            Jac, F = J # N x C x P, N x C
            vjp = torch.einsum('ncp,ncs->ps',Jac,v) # P x S
        else:
            projT = torch.vmap(self.vjp_single, (2,None,None))(v,self.params,x)
            vjp = [j.detach().flatten(1) for j in projT[0].values()]
            vjp = torch.cat(vjp,dim=1).detach().T # P x S
        return vjp
    

class ConjugateKernelSampler(object):
    def __init__(self, network, dtype):
        self.network = network
        self.device = next(network.parameters()).device
        self.dtype = dtype
        self.params = {k: v.detach() for k, v in self.network.named_parameters()} # dict of parameters

        self.params = tuple(self.network.parameters())
        child_list = list(self.network.children())
        if len(child_list) > 1:
            child_list = child_list
        elif len(child_list) == 1:
            child_list = child_list[0]
        first_layers = child_list[:-1]
        final_layer = child_list[-1]
        self.llp = list(final_layer.parameters())[0].size().numel()
        self.num_output = list(final_layer.parameters())[0].shape[0]
        self.features = torch.nn.Sequential(*first_layers)
        self.llp = list(self.features.parameters())[-1].shape[0]
        self.theta_t = util.flatten(self.params)[-(self.llp*self.num_output+self.num_output):-self.num_output]
    
    def sample_theta(self, S, scale):
        return torch.randn(size=(self.llp*self.num_output,S),device=self.device,dtype=self.dtype)*scale + self.theta_t.unsqueeze(1)   
    
    def vjp_single(self,v,params,x):
        _, vjp_fn = torch.func.vjp(lambda param: self.fnet(param, x), params)
        vjp = vjp_fn(v)
        return vjp
    
    def compute_full_jacobian(self, x):
        J = self.ck(x).detach() # N x LLP
        F = self.network(x).detach()
        return (J, F)
    
    def jvp(self, x, theta, J):
        '''
        x: input data, N x D
        theta: parameters, P x S
        J = None or (Jac, F) where Jacobian is N x C x P and F is N x C, with N = len(training data), C = number of classes, P = number of parameters. 
        Note: J is expected to be detached from computation graph.
        '''
        if J is not None:
            Jac, F = J # N x LLP, N x C
        else:
            Jac = self.features(x).detach() # N x LLP
            F = self.network(x).detach() # N x C

        V = (theta - self.theta_t.unsqueeze(1)).reshape(self.num_output, self.llp, -1).permute(1,0,2) # C*LLP x S -> C x LLP x S -> LLP x C x S
        proj = torch.einsum('np,pcs->ncs', Jac,V).detach() # N x C x S
        return F, proj # N x C, N x C x S

    def vjp(self, x, V, J):
        # V : x.shape[0] x self.n_output x self.S
        if J is not None:
            Jac, F = J # N x LLP, N x C

        else:
            Jac = self.features(x).detach() # N x LLP

        return torch.einsum('pn,ncs->pcs',Jac.T,V).detach().permute(1,0,2).flatten(0,1) # LLP x C x S -> C x LLP x S -> C*LLP x S
    
    def compute_flin(self, x, theta, J):
        f_nlin, proj = self.jvp(x, theta, J) # N x C, N x C x S
        return f_nlin.unsqueeze(2) + proj # N x C x S
    

    