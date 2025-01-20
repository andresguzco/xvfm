import torch
from torch import Tensor
from typing import List, Union, Type

class MLP(torch.nn.Module):
    def __init__(self, dim, out_dim=None, w=64):
        super().__init__()
        if out_dim is None:
            out_dim = dim
        self.net = torch.nn.Sequential(
            torch.nn.Linear(dim + 1, w),
            torch.nn.SELU(),
            torch.nn.Linear(w, w),
            torch.nn.SELU(),
            torch.nn.Linear(w, w),
            torch.nn.SELU(),
            torch.nn.Linear(w, out_dim),
        )

    def forward(self, x):
        return self.net(x)


class TabMLP(torch.nn.Module):

    class Block(torch.nn.Module):
        def __init__(
            self,
            *,
            d_in: int,
            d_out: int,
            dropout: float,
        ) -> None:
            super().__init__()
            self.linear = torch.nn.Linear(d_in, d_out, True)
            self.activation = torch.nn.ReLU()
            self.dropout = torch.nn.Dropout(dropout)

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
        classes: list
    ) -> None:
        """
        Note:
            `make_baseline` is the recommended constructor.
        """
        super().__init__()
        if isinstance(dropouts, float):
            dropouts = [dropouts] * len(d_layers)
        assert len(d_layers) == len(dropouts)

        self.num_feat = num_feat
        self.blocks = torch.nn.ModuleList(
            [
                TabMLP.Block(
                    d_in=d_layers[i - 1] if i else d_in,
                    d_out=d,
                    dropout=dropout,
                )
                for i, (d, dropout) in enumerate(zip(d_layers, dropouts))
            ]
        )
        self.head = torch.nn.Linear(d_layers[-1] if d_layers else d_in, d_out)
        self.filter = torch.nn.Sigmoid()
        self.classes = classes

    @classmethod
    def make_baseline(
        cls: Type['MLP'],
        d_in: int,
        d_layers: List[int],
        dropout: float,
        d_out: int,
        num_feat: int,
        classes: list
    ) -> 'MLP':

        assert isinstance(dropout, float)
        if len(d_layers) > 2:
            assert len(set(d_layers[1:-1])) == 1, (
                'if d_layers contains more than two elements, then'
                ' all elements except for the first and the last ones must be equal.'
            )
        return TabMLP(
            d_in=d_in,
            d_layers=d_layers,  # type: ignore
            dropouts=dropout,
            d_out=d_out,
            num_feat=num_feat,
            classes=classes
        )

    def forward(self, x: Tensor) -> Tensor:
        x = x.float()
        for block in self.blocks:
            x = block(x)
        x = self.head(x)
        
        res = torch.zeros(x.shape, device=x.device)
        res[:, :self.num_feat] = torch.nn.functional.normalize(x[:, :self.num_feat])
        # res[:, :self.num_feat] = x[:, :self.num_feat]
        cum_sum = self.num_feat
        for val in self.classes:
            filtered = torch.exp(torch.nn.functional.log_softmax(x[:, cum_sum:cum_sum + val], dim=1))
            res[:, cum_sum:cum_sum + val] = filtered / filtered.sum(dim=1, keepdim=True)
            cum_sum += val
        return res


class MultiMLP(torch.nn.Module):
    def __init__(self, d_in, classes, d_layers, dropout, is_y_cond, dim_t, num_feat):
        super().__init__()
        self.dim_t = dim_t
        self.classes = classes
        self.num_classes = len(classes)
        self.cond = is_y_cond
        self.mlp = TabMLP.make_baseline(
            d_in=dim_t, 
            d_layers=d_layers, 
            dropout=dropout, 
            d_out=d_in, 
            num_feat=num_feat,
            classes=classes
        )

        if self.num_classes > 0:
            self.label_emb = torch.nn.Embedding(self.num_classes, dim_t)
        elif self.num_classes == 0:
            self.label_emb = torch.nn.Linear(1, dim_t)
        
        self.proj = torch.nn.Linear(d_in, dim_t)
        self.time_embed = torch.nn.Sequential(
            torch.nn.Linear(1, dim_t),
            torch.nn.SiLU(),
            torch.nn.Linear(dim_t, dim_t)
        )
    
    def forward(self, x, t, y=None):
        emb = self.time_embed(t)
        if self.cond and y is not None:
            if self.num_classes > 0:
                y = y.squeeze()
            else:
                y = y.resize(y.size(0), 1).float()
            emb += F.silu(self.label_emb(y))
        x = self.proj(x) + emb
        return self.mlp(x)

