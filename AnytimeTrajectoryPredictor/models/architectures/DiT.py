import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def _get(config, key, default=None):
    return getattr(config, key, default) if config is not None else default


def modulate(x, shift, scale):
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


#################################################################################
#                          Diffusion Timestep Embedding                         #
#################################################################################

class TimestepEmbedder(nn.Module):
    """
    Embeds scalar timesteps into vector representations.
    """
    def __init__(self, hidden_size, frequency_embedding_size=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def timestep_embedding(t, dim, max_period=10000):
        """
        Create sinusoidal timestep embeddings.
        :param t: a 1-D Tensor of N indices, one per batch element.
                          These may be fractional.
        :param dim: the dimension of the output.
        :param max_period: controls the minimum frequency of the embeddings.
        :return: an (N, D) Tensor of positional embeddings.
        """
        # https://github.com/openai/glide-text2im/blob/main/glide_text2im/nn.py
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
        ).to(device=t.device)
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding

    def forward(self, t):
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
        t_emb = self.mlp(t_freq)
        return t_emb


#################################################################################
#                                 Core DiT Model                                 #
#################################################################################
class mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

class Attention(nn.Module):
    """
    Multi-head attention backed by PyTorch SDPA.

    Args:
        dim: Transformer dimension
        head_dim: Dimension of each attention head
        qkv_bias: Whether to include bias in the QKV linear layers
        proj_bias: Whether to include bias in the attention output projection
    """
    def __init__(
        self,
        dim: int,
        head_dim: int = 64,
        qkv_bias: bool = False,
        proj_bias: bool = False,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.num_heads = dim // head_dim
        self.head_dim = head_dim
        self.dropout = dropout

        self.Q = nn.Linear(dim, dim, bias=qkv_bias)
        self.K = nn.Linear(dim, dim, bias=qkv_bias)
        self.V = nn.Linear(dim, dim, bias=qkv_bias)

        self.attn_out_proj = nn.Linear(dim, dim, bias=proj_bias)

    def _split_heads(self, x):
        batch, sequence, channels = x.shape
        return x.view(batch, sequence, self.num_heads, self.head_dim).transpose(1, 2)

    def forward(self, x: torch.Tensor, context=None, mask=None) -> torch.Tensor:
        batch, sequence, channels = x.shape
        context = x if context is None else context
        q = self._split_heads(self.Q(x))
        k = self._split_heads(self.K(context))
        v = self._split_heads(self.V(context))
        if mask is not None:
            mask = mask[:, None]

        # SDPA dispatches to Flash Attention on supported CUDA inputs.
        x = F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=mask,
            dropout_p=self.dropout if self.training else 0.0,
        )
        x = x.transpose(1, 2).reshape(batch, sequence, channels)
        x = self.attn_out_proj(x)
        return x
    
class DiTBlock(nn.Module):
    """
    A DiT block with adaptive layer norm zero (adaLN-Zero) conditioning.
    """
    def __init__(self, hidden_size, num_heads, mlp_ratio=4.0, dropout=0.0, **block_kwargs):
        super().__init__()
        if hidden_size % num_heads != 0:
            raise ValueError("hidden_size must be divisible by num_heads")

        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        head_dim = hidden_size // num_heads
        self.attn = Attention(
            hidden_size,
            head_dim=head_dim,
            qkv_bias=True,
            dropout=dropout,
            **block_kwargs,
        )
        self.cross_norm = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.cross_attn = Attention(
            hidden_size,
            head_dim=head_dim,
            qkv_bias=True,
            dropout=dropout,
        )
        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        approx_gelu = lambda: nn.GELU(approximate="tanh")
        self.mlp = mlp(
            in_features=hidden_size,
            hidden_features=mlp_hidden_dim,
            act_layer=approx_gelu,
            drop=dropout,
        )
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 6 * hidden_size, bias=True)
        )

    def forward(self, x, c, context):
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(c).chunk(6, dim=1)
        x = x + gate_msa.unsqueeze(1) * self.attn(modulate(self.norm1(x), shift_msa, scale_msa))
        x = x + self.cross_attn(
            self.cross_norm(x),
            context=context,
        )
        x = x + gate_mlp.unsqueeze(1) * self.mlp(modulate(self.norm2(x), shift_mlp, scale_mlp))
        return x


class FinalLayer(nn.Module):
    """
    The final layer of DiT.
    """
    def __init__(self, hidden_size, output_size):
        super().__init__()
        self.norm_final = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(hidden_size, output_size, bias=True)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 2 * hidden_size, bias=True)
        )

    def forward(self, x, c):
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=1)
        hidden = modulate(self.norm_final(x), shift, scale)
        return self.linear(hidden), hidden


class DiT(nn.Module):
    """
    Diffusion Transformer adapted to ASTRA's polynomial trajectory tokens.
    """
    def __init__(self, config):
        super().__init__()
        self.num_modes = _get(
            config,
            "num_modes_K",
            _get(config, "num_modes", _get(config, "num_trajectory_possibilities", 6)),
        )
        self.trajectory_dim = _get(
            config,
            "trajectory_dim_dy",
            _get(config, "trajectory_dim", 2),
        )
        self.hidden_size = _get(config, "d_model", _get(config, "hidden_size", 256))
        self.max_agents = _get(config, "max_agents", 1)
        self.future_horizon = _get(
            config,
            "future_horizon_T",
            _get(config, "future_horizon", 80),
        )
        depth = _get(config, "dit_depth", _get(config, "num_layers", 4))
        num_heads = _get(config, "dit_num_heads", _get(config, "num_heads", 8))
        mlp_ratio = _get(config, "dit_mlp_ratio", _get(config, "mlp_ratio", 4.0))
        dropout = _get(config, "dropout", 0.0)

        self.trajectory_embedder = nn.Linear(self.trajectory_dim, self.hidden_size)
        self.mode_embedding = nn.Embedding(self.num_modes, self.hidden_size)
        self.agent_embedding = nn.Embedding(self.max_agents, self.hidden_size)
        self.future_time_embedding = nn.Embedding(self.future_horizon, self.hidden_size)
        self.t_embedder = TimestepEmbedder(self.hidden_size)
        self.context_norm = nn.LayerNorm(self.hidden_size)

        self.blocks = nn.ModuleList([
            DiTBlock(
                self.hidden_size,
                num_heads,
                mlp_ratio=mlp_ratio,
                dropout=dropout,
            )
            for _ in range(depth)
        ])
        self.final_layer = FinalLayer(self.hidden_size, self.trajectory_dim)
        self.initialize_weights()

    def initialize_weights(self):
        # Initialize transformer layers:
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
        self.apply(_basic_init)
        
        # Initialize timestep embedding MLP:
        nn.init.normal_(self.t_embedder.mlp[0].weight, std=0.02)
        nn.init.normal_(self.t_embedder.mlp[2].weight, std=0.02)

        # Zero-out adaLN modulation layers in DiT blocks:
        for block in self.blocks:
            nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.adaLN_modulation[-1].bias, 0)

        # Zero-out output layers:
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].bias, 0)
        nn.init.constant_(self.final_layer.linear.weight, 0)
        nn.init.constant_(self.final_layer.linear.bias, 0)

    def forward(self, x, context, noise_level):
        """
        Denoise ASTRA coefficient tokens.

        x: (batch, modes, agents, future_horizon, trajectory_dim)
        context: (batch, context_tokens, hidden_size)
        noise_level: one scalar EDM noise value per batch item
        """
        if x.dim() != 5:
            raise ValueError("x must have shape (B, K, N, T, D)")
        if context.dim() != 3 or context.shape[-1] != self.hidden_size:
            raise ValueError(
                f"context must have shape (B, tokens, {self.hidden_size})"
            )

        batch, modes, agents, horizon, trajectory_dim = x.shape
        if trajectory_dim != self.trajectory_dim:
            raise ValueError(
                f"x trajectory dimension must be {self.trajectory_dim}, got {trajectory_dim}"
            )
        if context.shape[0] != batch:
            raise ValueError("x and context must have the same batch size")
        if modes > self.num_modes:
            raise ValueError(f"x contains {modes} modes, configured for {self.num_modes}")
        if agents > self.max_agents:
            raise ValueError(
                f"x contains {agents} agents, configured for {self.max_agents}"
            )
        if horizon > self.future_horizon:
            raise ValueError(
                f"x future horizon is {horizon}, configured for {self.future_horizon}"
            )

        mode_ids = torch.arange(modes, device=x.device).view(1, modes, 1, 1)
        agent_ids = torch.arange(agents, device=x.device).view(1, 1, agents, 1)
        time_ids = torch.arange(horizon, device=x.device).view(1, 1, 1, horizon)
        tokens = (
            self.trajectory_embedder(x)
            + self.mode_embedding(mode_ids)
            + self.agent_embedding(agent_ids)
            + self.future_time_embedding(time_ids)
        )
        tokens = tokens.reshape(batch, modes * agents * horizon, self.hidden_size)

        noise_level = noise_level.reshape(-1)
        if noise_level.numel() == 1 and batch > 1:
            noise_level = noise_level.expand(batch)
        if noise_level.numel() != batch:
            raise ValueError("noise_level must have one scalar per batch item")
        c = self.t_embedder(noise_level)
        context = self.context_norm(context)

        for block in self.blocks:
            tokens = block(tokens, c, context)

        denoised, hidden = self.final_layer(tokens, c)
        return (
            denoised.reshape(batch, modes, agents, horizon, self.trajectory_dim),
            hidden.reshape(batch, modes, agents, horizon, self.hidden_size),
        )

