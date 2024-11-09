import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.rms import RMSNorm


class MambaResidual(nn.Module):
    def __init__(self):
        super(MambaResidual, self).__init__()

        self.model = Mamba()
        self.norm = RMSNorm(dim=64)

    def forward(self, x):
        '''
        a basic residual block with the following operations:
        LayerNorm -> Mamba -> Residual

        x : (B, L, D)
        y : (B, L, D)
        '''
        y = x + self.model(self.norm(x))
        return y
    

class Mamba(nn.Module):
    def __init__(self,
                 d_model=64,
                 d_state=16,
                 expansion=2,
                 dt_rank=4,
                 conv_kernel=4,
                 dt_softplus=True,
                 ):
        '''a singular mamba block as introduced in https://arxiv.org/abs/2312.00752'''
        super(Mamba, self).__init__()

        self.d_model = d_model
        self.expansion = expansion
        self.d_inner = int(self.d_model * self.expansion)
        self.d_state = d_state
        self.dt_rank = dt_rank
        self.conv_kernel = conv_kernel
        self.dt_softplus = dt_softplus

        self.in_proj = nn.Linear(self.d_model, self.d_inner*2) 

        self.conv1d = nn.Conv1d(in_channels=self.d_inner,
                              out_channels=self.d_inner,
                              kernel_size=self.conv_kernel,
                              padding=self.conv_kernel-1)
        
        self.act = nn.SiLU()
        self.x_proj = nn.Linear(self.d_inner, self.dt_rank + self.d_state*2)
        self.dt_proj = nn.Linear(self.dt_rank, self.d_inner)
        self.out_proj = nn.Linear(self.d_inner, self.d_model)

        A = torch.arange(1, self.d_state+1, dtype=torch.float32).repeat(self.d_inner, 1).contiguous()
        self.A_log = nn.Parameter(torch.log(A))
        self.D = nn.Parameter(torch.ones(self.d_inner))

    def forward(self, x):
        '''
        mamba is a seq2seq model
        x   : (B,L,D)
        out : (B,L,D)
        '''

        _, l, _ = x.shape # (B L D)

        # split input x and skip connection
        xz = self.in_proj(x) # (B, L, 2ED)
        x, z = xz.chunk(2, dim=-1) # (B, L, ED) each
        z = self.act(z) 

        # x is padded along l dim, so slice it upto initial length
        x = self.conv1d(x.transpose(1,2).contiguous())[:,:,:l]
        x = x.transpose(1,2).contiguous()
        x = self.act(x)

        # delta (B, L, dt_rank) -> (B, L, ED), B and C (B, L, N)
        x_proj = self.x_proj(x)
        delta, B, C = x_proj.split([self.dt_rank,  self.d_state, self.d_state], dim=-1)
        delta = self.dt_proj(delta)

        if self.dt_softplus:
            delta = F.softplus(delta)

        # matrices A,B,C,D for system state equations
        A = -torch.exp(self.A_log.float()).to(x.device)
        D = self.D.float().to(x.device)
        B = B.float()
        C = C.float()

        # discretize A and B matrices
        A_bar = torch.exp(torch.einsum('bld,dn->bldn', delta, A)) # (B, L, ED, N)
        B_bar_x = torch.einsum('bld,bln,bld->bldn', delta, B, x) # (B, L, ED, N) / B_bar*x(t)

        b, l, d_in, n = A_bar.shape

        # hidden state (B, ED, N)
        h = torch.zeros(b,d_in,n, device=x.device)
        y = []
        # propagate through seqlen dim to compute h and y
        for i in range(l):
            # h(t) = A*h(t-1) + B*x(t)
            h = A_bar[:,i] * h + B_bar_x[:,i]
            # y(t) = C*h(t)
            y_ = torch.einsum('bdn,bn->bd', h, C[:,i])
            y.append(y_)     
        y = torch.stack(y,dim=1)

        # skip connection, y(t) = C*h(t) + D*x(t)
        y = y + D*x

        if z is not None:
            y = y*z
        
        # (B, L, ED) --> (B, L, D)
        out = self.out_proj(y)
        return out