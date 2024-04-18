from typing import List

from equinox import field
import equinox as eqx
import equinox.nn as nn
from einops import rearrange, repeat

import jax
import jax.numpy as jnp
from jaxtyping import Array

from .layers import WNConv1d

import time


# TODO consolidate
def pseudo_rn():
    return jax.random.PRNGKey(int(time.perf_counter()))


def _normalize(x: Array) -> Array:
    return x / repeat(
        jnp.clip(jnp.linalg.norm(x, axis=-1), 1e-12), "n -> n r", r=x.shape[-1]
    )


class VectorQuantize(eqx.Module):
    codebook_size: int = field(static=True)
    codebook_dim: int = field(static=True)
    stride: int = field(static=True)

    in_proj: nn.WeightNorm
    out_proj: nn.WeightNorm
    codebook: nn.Embedding
    pool: nn.AvgPool1d

    def __init__(
        self, input_dim: int, codebook_size: int, codebook_dim: int, stride: int = 1
    ):
        super().__init__()
        self.codebook_size = codebook_size
        self.codebook_dim = codebook_dim
        self.stride = stride

        self.in_proj = WNConv1d(input_dim, codebook_dim, kernel_size=1)
        self.out_proj = WNConv1d(codebook_dim, input_dim, kernel_size=1)
        self.codebook = nn.Embedding(codebook_size, codebook_dim, key=pseudo_rn())

        self.pool = nn.AvgPool1d(stride, stride)

    def __call__(self, z):
        if self.stride > 1:
            z = self.pool(z)

        # Factorized codes (ViT-VQGAN) Project input into low-dimensional space
        z_e = self.in_proj(z)  # z_e : (B x D x T)
        z_q, indices = self.decode_latents(z_e)
        z_q = z_e + jax.lax.stop_gradient(
            z_q - z_e
        )  # noop in forward pass, straight-through gradient estimator in backward pass

        z_q = self.out_proj(z_q)

        if self.stride > 1:
            z_q = jnp.repeat(z_q, self.stride, axis=-1)

        return z_q, indices

    def embed_code(self, embed_id):
        return jnp.vectorize(self.codebook, signature="()->(n)")(embed_id)

    def decode_code(self, embed_id):
        return self.embed_code(embed_id).T  # TODO tranpose?

    def decode_latents(self, latents):
        encodings = rearrange(latents, "d t -> t d")
        codebook = self.codebook.weight  # codebook: (N x D)

        # L2 normalize encodings and codebook (ViT-VQGAN)
        encodings = _normalize(encodings)
        codebook = _normalize(codebook)

        # Compute euclidean distance with codebook
        term1 = jnp.sum(jnp.square(encodings), axis=1, keepdims=True)
        term2 = 2 * encodings @ codebook.T
        term3 = jnp.sum(jnp.square(codebook), axis=1, keepdims=True).T

        dist = term1 - term2 + term3
        indices = jnp.argmax(-dist, axis=1)
        z_q = self.decode_code(indices)
        return z_q, indices


class ResidualVectorQuantize(eqx.Module):
    n_codebooks: int = field(static=True)
    codebook_dim: int = field(static=True)
    codebook_size: int = field(static=True)
    quantizers: list[VectorQuantize]

    def __init__(
        self,
        input_dim: int = 512,
        codebook_size: int = 1024,
        codebook_dim: int = 8,
        vq_strides: List[int] = [1, 1, 1, 1],
    ):
        self.n_codebooks = len(vq_strides)
        self.codebook_dim = codebook_dim
        self.codebook_size = codebook_size
        self.quantizers = [
            VectorQuantize(input_dim, codebook_size, codebook_dim, stride)
            for stride in vq_strides
        ]

    def __call__(self, z):
        z_q = 0.0
        residual = z
        codes = []
        for i, quantizer in enumerate(self.quantizers):
            z_q_i, indices_i = quantizer(residual)
            z_q = z_q + z_q_i
            residual = residual - z_q_i
            codes.append(indices_i)

        return z_q, codes

    def from_codes(self, codes: List[Array]) -> Array:
        z_q = 0.0
        for i in range(self.n_codebooks):
            z_p_i = self.quantizers[i].decode_code(codes[i])
            z_q_i = self.quantizers[i].out_proj(z_p_i)
            # z_q_i = z_q_i.repeat_interleave(self.quantizers[i].stride, dim=-1)
            z_q_i = jnp.repeat(z_q_i, self.quantizers[i].stride, axis=-1)
            z_q += z_q_i
        return z_q
