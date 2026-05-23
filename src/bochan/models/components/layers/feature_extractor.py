import torch
import torch.nn as nn
from typing import Optional


def _make_activation(name: str) -> nn.Module:
    """
    活性化関数を名前から生成する。
    """
    name = name.lower()
    if name == "relu":
        return nn.ReLU()
    if name == "leaky_relu":
        return nn.LeakyReLU()
    if name == "gelu":
        return nn.GELU()
    if name == "elu":
        return nn.ELU()
    raise ValueError(f"Unknown activation: {name}")


class LargeFeatureExtractor(nn.Module):
    """
    Deep Kernel Learning 用の標準的な MLP 特徴抽出器。

    元の実装:
        input_dim -> input_dim*10 -> input_dim*5 -> input_dim*2 -> output_dim

    をベースにしつつ、hidden_dims や活性化関数などを柔軟に変更できるようにした版。
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_dims: Optional[list[int]] = None,
        activation: str = "leaky_relu",
        dropout: float = 0.0,
        use_bn: bool = False,
    ) -> None:
        """
        Args:
            input_dim (int): 入力次元
            output_dim (int): 出力次元
            hidden_dims (Optional[list[int]]): 隠れ層の次元
            activation (str): 活性化関数名
            dropout (float): Dropout率
            use_bn (bool): BatchNormを使うかどうか
        """
        super().__init__()

        if hidden_dims is None:
            hidden_dims = [input_dim * 10, input_dim * 5, input_dim * 2]
            # hidden_dims = [input_dim * 5, input_dim * 2]

        if len(hidden_dims) == 0:
            raise ValueError("hidden_dims には少なくとも1つの要素が必要です。")

        act = _make_activation(activation)

        dims = [input_dim] + hidden_dims
        layers: list[nn.Module] = []

        for in_dim, out_dim in zip(dims[:-1], dims[1:]):
            layers.append(nn.Linear(in_dim, out_dim))
            if use_bn:
                layers.append(nn.BatchNorm1d(out_dim))
            layers.append(_make_activation(activation))
            if dropout > 0:
                layers.append(nn.Dropout(dropout))

        layers.append(nn.Linear(dims[-1], output_dim))

        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): shape = (..., input_dim)

        Returns:
            torch.Tensor: shape = (..., output_dim)
        """
        return self.network(x)

class ResidualLinearBlock(nn.Module):
    """
    線形層 + 活性化 + skip connection のブロック。

    入出力次元が異なる場合は skip 側を線形射影して加算する。
    """

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        activation: str = "leaky_relu",
        dropout: float = 0.0,
        use_bn: bool = False,
    ) -> None:
        """
        Args:
            in_dim (int): 入力次元
            out_dim (int): 出力次元
            activation (str): 活性化関数名
            dropout (float): Dropout率
            use_bn (bool): BatchNormを使うかどうか
        """
        super().__init__()

        layers: list[nn.Module] = [nn.Linear(in_dim, out_dim)]
        if use_bn:
            layers.append(nn.BatchNorm1d(out_dim))
        layers.append(_make_activation(activation))
        if dropout > 0:
            layers.append(nn.Dropout(dropout))

        self.main = nn.Sequential(*layers)
        self.skip = nn.Identity() if in_dim == out_dim else nn.Linear(in_dim, out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): shape = (..., in_dim)

        Returns:
            torch.Tensor: shape = (..., out_dim)
        """
        return self.main(x) + self.skip(x)


class SkipLargeFeatureExtractor(nn.Module):
    """
    skip / residual 構造付きの特徴抽出器。

    元の LargeFeatureExtractor と同様に、
        input_dim -> input_dim*10 -> input_dim*5 -> input_dim*2 -> output_dim
    の流れを保ちつつ、各段に residual connection を入れる。
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_dims: Optional[list[int]] = None,
        activation: str = "leaky_relu",
        dropout: float = 0.0,
        use_bn: bool = False,
        use_global_skip: bool = True,
    ) -> None:
        """
        Args:
            input_dim (int): 入力次元
            output_dim (int): 出力次元
            hidden_dims (Optional[list[int]]): 隠れ層の次元
            activation (str): 活性化関数名
            dropout (float): Dropout率
            use_bn (bool): BatchNormを使うかどうか
            use_global_skip (bool): 入力から最終出力への global skip を入れるか
        """
        super().__init__()

        if hidden_dims is None:
            hidden_dims = [input_dim * 10, input_dim * 5, input_dim * 2]

        if len(hidden_dims) == 0:
            raise ValueError("hidden_dims には少なくとも1つの要素が必要です。")

        dims = [input_dim] + hidden_dims

        self.blocks = nn.ModuleList([
            ResidualLinearBlock(
                in_dim=dims[i],
                out_dim=dims[i + 1],
                activation=activation,
                dropout=dropout,
                use_bn=use_bn,
            )
            for i in range(len(dims) - 1)
        ])

        self.final_linear = nn.Linear(dims[-1], output_dim)

        self.use_global_skip = use_global_skip
        if use_global_skip:
            self.global_skip = (
                nn.Identity() if input_dim == output_dim else nn.Linear(input_dim, output_dim)
            )
        else:
            self.global_skip = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): shape = (..., input_dim)

        Returns:
            torch.Tensor: shape = (..., output_dim)
        """
        x0 = x
        h = x

        for block in self.blocks:
            h = block(h)

        out = self.final_linear(h)

        if self.global_skip is not None:
            out = out + self.global_skip(x0)

        return out