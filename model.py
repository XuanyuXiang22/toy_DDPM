import torch
import torch.nn as nn


class MLPDiffusion(nn.Module):
    def __init__(self, opt):
        super(MLPDiffusion, self).__init__()
        self.opt = opt

        self.linears = nn.ModuleList([
            nn.Linear(2, self.opt.num_uints),
            nn.ReLU(),
            nn.Linear(self.opt.num_uints, self.opt.num_uints),
            nn.ReLU(),
            nn.Linear(self.opt.num_uints, self.opt.num_uints),
            nn.ReLU(),
            nn.Linear(self.opt.num_uints, 2),
        ])
        self.step_embeddings = nn.ModuleList([
            nn.Embedding(self.opt.num_steps, self.opt.num_uints),
            nn.Embedding(self.opt.num_steps, self.opt.num_uints),
            nn.Embedding(self.opt.num_steps, self.opt.num_uints),
        ])

    def forward(self, x_t, t):
        x = x_t
        for idx, embedding_layer in enumerate(self.step_embeddings):
            t_embedding = embedding_layer(t)
            x = self.linears[2*idx](x)
            x += t_embedding
            x = self.linears[2*idx+1](x)
        x = self.linears[-1](x)

        return x