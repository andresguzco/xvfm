import torch.nn as nn
from torch.nn.functional import log_softmax

from torch import Tensor, exp, zeros, zeros_like
from typing import List, Union, Type


class MLP(nn.Module):

    class Block(nn.Module):
        def __init__(
            self,
            *,
            d_in: int,
            d_out: int,
            dropout: float,
        ) -> None:
            super().__init__()
            self.linear = nn.Linear(d_in, d_out, True)
            self.activation = nn.ReLU()
            self.dropout = nn.Dropout(dropout)

        def forward(self, x: Tensor) -> Tensor:
            return self.dropout(self.activation(self.linear(x)))

    def __init__(
        self,
        *,
        d_in: int,
        d_layers: List[int],
        dropouts: Union[float, List[float]],
        d_out: int,
        num_feat: int,
        classes: list,
        task:str
    ) -> None: 
        super().__init__()
        dropouts = [dropouts] * len(d_layers)
        assert len(d_layers) == len(dropouts)

        self.num_feat = num_feat
        self.blocks = nn.ModuleList(
            [
                MLP.Block(
                    d_in=d_layers[i - 1] if i else d_in,
                    d_out=d,
                    dropout=dropout,
                )
                for i, (d, dropout) in enumerate(zip(d_layers, dropouts))
            ]
        )
        self.head = nn.Linear(d_layers[-1] if d_layers else d_in, d_out)
        self.classes = classes
        self.task = task

    @classmethod
    def make_baseline(
        cls: Type['MLP'],
        d_in: int,
        d_layers: List[int],
        dropout: float,
        d_out: int,
        num_feat: int,
        classes: list,
        task: str
    ) -> 'MLP':

        if len(d_layers) > 2:
            assert len(set(d_layers[1:-1])) == 1, (
                'if d_layers contains more than two elements, then'
                ' all elements except for the first and the last ones must be equal.'
            )
        return MLP(
            d_in=d_in,
            d_layers=d_layers,
            dropouts=dropout,
            d_out=d_out,
            num_feat=num_feat,
            classes=classes,
            task=task
        )

    def forward(self, x: Tensor) -> Tensor:
        x = x.float()

        for block in self.blocks:
            x = block(x)

        x = self.head(x)
        
        res = zeros(x.shape, device=x.device)
        cum_sum = self.num_feat
        res[:, :cum_sum] = x[:, :cum_sum]
        
        if sum(self.classes) > 0:
            for val in self.classes:
                res[:, cum_sum:cum_sum + val] = exp(log_softmax(x[:, cum_sum:cum_sum + val], dim=1))
                cum_sum += val

        if self.task == 'regression':
            res[:, -1] = x[:, -1]
        else:
            res[:, -1] = exp(log_softmax(x[:, -1], dim=0))

        return res


class MultiMLP(nn.Module):
    def __init__(self, d_in, classes, num_feat, task):
        super().__init__()
        self.num_feat = num_feat
        self.classes = classes
        d_layers = [512, 512, 512, 512]
        dim_t = 128
        
        self.mlp = MLP.make_baseline(
            d_in=dim_t, 
            d_layers=d_layers, 
            dropout=0.1, 
            d_out=d_in, 
            num_feat=num_feat,
            classes=classes,
            task=task
        )

        self.proj = nn.Linear(d_in, dim_t)
        self.time_embed = nn.Sequential(
            nn.Linear(1, dim_t),
            nn.SiLU(),
            nn.Linear(dim_t, dim_t)
        )

    def parameters(self):        
        return list(self.mlp.parameters()) + list(self.proj.parameters()) + list(self.time_embed.parameters())

    def forward(self, x, t):
        emb = self.time_embed(t)
        x_emb = self.proj(x) + emb
        return self.mlp(x_emb)


class Tabformer(nn.Module):
    def __init__(self, d_in, classes, num_feat, task, d_layer=4, n_layer=4, dropout=0.1, dim_t=128):
        super().__init__()
        self.num_feat = num_feat
        self.classes = classes
        self.task = task

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=dim_t,
            nhead=d_layer,
            dropout=dropout,
            batch_first=True 
        )
        self.encoder = nn.TransformerEncoder(
            encoder_layer, 
            num_layers=n_layer
        )
        self.input_proj = nn.Linear(d_in, dim_t)
        self.time_embed = nn.Sequential(
            nn.Linear(1, dim_t),
            nn.SiLU(),
            nn.Linear(dim_t, dim_t)
        )
        self.out_proj = nn.Linear(dim_t, d_in)

    def parameters(self):

        return list(self.encoder.parameters()) + list(self.input_proj.parameters()) + list(self.time_embed.parameters()) + list(self.out_proj.parameters())

    def forward(self, x, t):
        x = x.float()

        emb = self.input_proj(x).unsqueeze(1)
        t_emb = self.time_embed(t.float())
        emb += t_emb.unsqueeze(1)

        hidden = self.encoder(emb)
        hidden = hidden.squeeze(1)

        logits = self.out_proj(hidden)
        res = zeros_like(logits)
        
        cum_sum = self.num_feat
        res[:, :cum_sum] = logits[:, :cum_sum]
        
        if sum(self.classes) != 0:
            for val in self.classes:
                slice_logits = logits[:, cum_sum : cum_sum + val]
                cat_probs = exp(log_softmax(slice_logits, dim=1))
                res[:, cum_sum : cum_sum + val] = cat_probs 
                cum_sum += val

        if self.task == 'regression':
            res[:, -1] = logits[:, -1]
        else:
            res[:, -1] = exp(log_softmax(logits[:, -1], dim=0))

        return res
