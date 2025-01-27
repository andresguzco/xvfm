import torch.nn as nn
from torch.nn.functional import log_softmax, silu

from torch import Tensor, exp, zeros
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

        for val in self.classes:
            res[:, cum_sum:cum_sum + val] = exp(log_softmax(x[:, cum_sum:cum_sum + val], dim=1))
            cum_sum += val

        if self.task == 'regression':
            res[:, -1] = x[:, -1]
        else:
            res[:, -1] = exp(log_softmax(x[:, -1], dim=0))

        return res


class MultiMLP(nn.Module):
    def __init__(self, d_in, classes, d_layers, n_layers, dropout, dim_t, num_feat, task):
        super().__init__()
        # self.mlp = MLP.make_baseline(
        #     d_in=dim_t, 
        #     d_layers=d_layers, 
        #     dropout=dropout, 
        #     d_out=d_in + 1, 
        #     num_feat=num_feat,
        #     classes=classes,
        #     task=task
        # )

        self.label_emb = nn.Linear(1, dim_t)
        self.proj = nn.Linear(d_in + 1, dim_t)
        self.time_embed = nn.Sequential(
            nn.Linear(1, dim_t),
            nn.SiLU(),
            nn.Linear(dim_t, dim_t)
        )

    def forward(self, x, t, y=None):
        emb = self.time_embed(t)

        if y is not None:
            y_emb = self.label_emb(y.float())
            emb += silu(y_emb)

        x_emb = self.proj(x) + emb
        return self.mlp(x_emb, x)


class Tabformer(nn.Module):
    def __init__(self, d_in, classes, d_layers, n_layers, dropout, dim_t, num_feat, task):
        super().__init__()
        """
        d_in: number of input columns (features).
        classes, num_feat, task: not used here, but left in for consistency.
        d_layers: dimension for the Transformer feedforward layers.
        n_layers: number of Transformer encoder layers.
        dropout: dropout rate for Transformer.
        dim_t: embedding dimension (d_model) for Transformer.
        """

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=dim_t,
            nhead=4,
            dim_feedforward=d_layers,
            dropout=dropout,
            batch_first=False
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

        self.out_proj = nn.Linear(dim_t, 1)
        self.time_embed = nn.Sequential(
            nn.Linear(1, dim_t),
            nn.SiLU(),
            nn.Linear(dim_t, dim_t),
        )
        self.label_emb = nn.Linear(1, dim_t)  
        self.column_proj = nn.Linear(1, dim_t)

    def forward(self, x, t, y=None):
        """
        x: [batch_size, d_in]   (each row has d_in columns)
        t: [batch_size, 1]      (time or some scalar you want to embed)
        y: [batch_size, 1] or None (optional label/condition)
        Returns: [batch_size, d_in] (one scalar prediction per column)
        """

        # 1) Expand x so each column is a "token"
        #    x.shape => [batch_size, d_in, 1]
        x = x.unsqueeze(-1)

        # 2) Project each scalar column into a dim_t embedding => [batch_size, d_in, dim_t]
        x_emb = self.column_proj(x)

        # 3) Add time embedding (broadcast to all columns/tokens)
        t_emb = self.time_embed(t)                             # [batch_size, dim_t]
        t_emb = t_emb.unsqueeze(1).expand(-1, x.shape[1], -1)  # => [batch_size, d_in, dim_t]
        x_emb = x_emb + t_emb

        # 4) If label y is provided, embed it and add to all tokens
        if y is not None:
            y_emb = self.label_emb(y.float())                       # [batch_size, dim_t]
            y_emb = y_emb.unsqueeze(1).expand(-1, x.shape[1], -1)   # => [batch_size, d_in, dim_t]
            x_emb = x_emb + y_emb

        # 5) Transformer expects shape [seq_len, batch_size, d_model] (if batch_first=False)
        #    So transpose: [batch_size, d_in, dim_t] => [d_in, batch_size, dim_t]
        x_emb = x_emb.transpose(0, 1)

        # 6) Encode the sequence of columns
        hidden = self.encoder(x_emb)  # => [d_in, batch_size, dim_t]

        # 7) Map each token’s final embedding to a single scalar
        out = self.out_proj(hidden)   # => [d_in, batch_size, 1]

        # 8) Transpose back to [batch_size, d_in]
        out = out.transpose(0, 1)     # => [batch_size, d_in, 1]
        out = out.squeeze(-1)         # => [batch_size, d_in]

        return out
