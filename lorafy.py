import torch
from torch import nn
import einops
import numpy as np


class LoRALinear(nn.Module):
    def __init__(self,linear,dim):
        nn.Module.__init__(self)
        self.linear  = linear
        self.LoRAin  = nn.Linear(linear.in_features,dim,bias=False)
        self.LoRAout = nn.Linear(dim,linear.out_features,bias=False)

        with torch.no_grad():
            self.LoRAin.weight*=0.0001
            self.LoRAout.weight*=0.0001

    def forward(self,x):

        y = self.linear(x)

        if self.training:
            LoRAactivation = self.LoRAin(x)
            y += self.LoRAout(LoRAactivation)

        return y

    def train(self,mode):
        if self.training != mode:
            with torch.no_grad():
                self.linear.weight += (2*bool(self.training)-1) * self.LoRAout.weight.matmul(self.LoRAin.weight)
            self.training = not self.training
        return self

class MultiheadAttention(nn.Module):# equivalent to torch.nn.MultiheadAttention, but it does something weird so reimplemented it
    def __init__(self, d_model, n_head):
        super().__init__()
        self.n_head = n_head
        self.out_proj = nn.Linear(d_model, d_model)
        self.in_proj = nn.Linear(d_model, 3*d_model)

    def forward(self, x, *args,**kwargs):
        x=x.permute(1, 0, 2)
        qkv = self.in_proj(x)
        q, k, v = einops.rearrange(qkv,'n k (p h c) -> p n h k c', p=3, h=self.n_head)
        qk = torch.einsum('nhqc,nhkc->nhqk', q, k) / np.sqrt(q.shape[-1])
        qk = nn.functional.softmax(qk, dim=-1)
        out = torch.einsum('nhqk,nhkc->nhqc', qk, v)
        out = einops.rearrange(out,'n h k c -> n k (h c)')
        out = self.out_proj(out)
        out = out.permute(1, 0, 2)
        return out,qk

class DepthwiseSeparableConv(torch.nn.Module):
    def __init__(self, nin, nout, kernel_size, stride,bias=False):
        super(DepthwiseSeparableConv, self).__init__()
        self.depthwise = torch.nn.Conv2d(nin, nin, kernel_size=kernel_size, stride=stride, groups=nin, bias=False)
        self.pointwise = torch.nn.Conv2d(nin, nout, kernel_size=1, bias=bias)

    def forward(self, x):
        out = self.depthwise(x)
        out = self.pointwise(out)
        return out

def lorafy(module,dim):    
    for name,layer in module._modules.items():
        if isinstance(layer, nn.MultiheadAttention):
            module._modules[name] = MultiheadAttention(layer.embed_dim,layer.embed_dim//layer.head_dim)
            module._modules[name].out_proj = layer.out_proj
            module._modules[name].in_proj.weight = layer.in_proj_weight
            module._modules[name].in_proj.bias = layer.in_proj_bias
            layer = module._modules[name]
        lorafy(layer,dim)
        if isinstance(layer, nn.Linear):
            module._modules[name] = LoRALinear(layer,dim)

def loraweights(module,n=''):
    weights = {}
    for name,layer in module._modules.items():
        weights.update(loraweights(layer,f'{n}_{name}'))
        if isinstance(layer, LoRALinear):
            weights[f'{n}_{name}_in']=layer.LoRAin.weight
            weights[f'{n}_{name}_out']=layer.LoRAout.weight
    return weights

def loadlora(module,train=True,n='',w={}):
    for name,layer in module._modules.items():
        if train:
            if isinstance(layer, nn.MultiheadAttention):
                module._modules[name] = MultiheadAttention(layer.embed_dim,layer.embed_dim//layer.head_dim)
                module._modules[name].out_proj = layer.out_proj
                module._modules[name].in_proj.weight = layer.in_proj_weight
                module._modules[name].in_proj.bias = layer.in_proj_bias
                layer = module._modules[name]
            loadlora(layer,train,f'{n}_{name}',w)
            if isinstance(layer, nn.Linear):
                module._modules[name] = LoRALinear(layer,1)
                module._modules[name].LoRAin.weight.data = w[f'{n}_{name}_in']
                module._modules[name].LoRAout.weight.data = w[f'{n}_{name}_out']
        else:
            if isinstance(layer, nn.Linear):
                module._modules[name].weight += w[f'{n}_{name}'][1].matmul(w[f'{n}_{name}'][0])
            
