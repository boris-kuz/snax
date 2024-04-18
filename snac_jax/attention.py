from einops import rearrange
import equinox as eqx
from equinox import nn
from equinox import field
from equinox.nn._attention import dot_product_attention
from jaxtyping import Array
import jax.numpy as jnp
import jax
import time


# TODO consolidate
def pseudo_rn():
    return jax.random.PRNGKey(int(time.perf_counter()))


class SinusoidalEmbeddings(eqx.Module):
    inv_freq: Array
    scale: Array
    use_xpos: bool
    scale_base: float | None = field(static=True)

    def __init__(self, dim, scale_base=None, use_xpos=False):
        self.inv_freq = 1.0 / (
            10000 ** (jnp.arange(0, dim, 2).astype(float) / dim)
        )  # TODO buffer
        # xpos related
        self.use_xpos = use_xpos
        self.scale_base = scale_base
        assert not (
            use_xpos and scale_base is None
        ), "scale base must be defined if using xpos"
        self.scale = (jnp.arange(0, dim, 2) + 0.4 * dim) / (1.4 * dim)
        # self.register_buffer("scale", scale, persistent=False)

    def __call__(self, x, key=None):
        seq_len, device = x.shape[-2], x.device
        inv_freq = jax.lax.stop_gradient(self.inv_freq)
        t = jnp.arange(seq_len).astype(inv_freq)
        freqs = jnp.einsum("i , j -> i j", t, inv_freq)
        freqs = jnp.concatenate((freqs, freqs), axis=-1)
        if not self.use_xpos:
            return freqs, jnp.ones(1, device=device)
        power = (t - (seq_len // 2)) / self.scale_base
        scale = jax.lax.stop_gradient(self.scale) ** rearrange(power, "n -> n 1")
        scale = jnp.concatenate((scale, scale), axis=-1)

        return freqs, scale


class LocalMHA(eqx.Module):
    norm: nn.LayerNorm
    heads: int
    window_size: int
    to_qkv: nn.Linear
    rel_pos: SinusoidalEmbeddings | None
    to_out: nn.Linear

    def __init__(self, dim=1024, window_size=32, dim_head=64, use_rotary_pos_emb=True):
        self.norm = nn.LayerNorm(dim)
        self.heads = dim // dim_head
        self.window_size = window_size
        self.to_qkv = nn.Linear(dim, dim * 3, use_bias=False, key=pseudo_rn())
        if use_rotary_pos_emb:
            self.rel_pos = SinusoidalEmbeddings(dim_head, scale_base=window_size // 2)
        else:
            self.rel_pos = None
        self.to_out = nn.Linear(dim, dim, use_bias=False, key=pseudo_rn())

    def __call__(self, x, key=None):
        B, C, T = x.shape
        residual = x
        x = self.norm(x.transpose(1, 2))
        windows = T // self.window_size
        q, k, v = jnp.split(self.to_qkv(x), 3, axis=-1)
        q, k, v = map(
            lambda t: rearrange(
                t, "b (w n) (h d) -> b h w n d", w=windows, h=self.heads
            ),
            (q, k, v),
        )
        if self.rel_pos is not None:
            pos_emb, scale = self.rel_pos(k)
            q, k = apply_rotary_pos_emb(q, k, pos_emb, scale)
        out = dot_product_attention(q, k, v)
        out = rearrange(out, "b h w n d -> b (w n) (h d)")
        out = self.to_out(out)
        return out.transpose(1, 2) + residual


def rotate_half(x):
    x = rearrange(x, "b ... (r d) -> b ... r d", r=2)
    # x1, x2 = x.unbind(dim=-2)
    x1, x2 = [s.squeeze() for s in jnp.split(x, x.shape[-2], -2)]
    return jnp.concatenate((-x2, x1), axis=-1)


def apply_rotary_pos_emb(q, k, freqs, scale):
    q_len = q.shape[-2]
    q_freqs = freqs[..., -q_len:, :]
    inv_scale = scale**-1
    if scale.ndim == 2:
        scale = scale[-q_len:, :]
    q = (q * jnp.cos(q_freqs) * scale) + (rotate_half(q) * jnp.sin(q_freqs) * scale)
    k = (k * jnp.cos(freqs) * inv_scale) + (rotate_half(k) * jnp.sin(freqs) * inv_scale)
    return q, k
