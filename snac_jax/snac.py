import json
import math
from typing import List, Tuple

import numpy as np

import jax
import jax.numpy as jnp
import equinox as eqx
from equinox import nn
from equinox import field

from jaxtyping import Array

from .layers import Encoder, Decoder
from .vq import ResidualVectorQuantize

from loguru import logger


class SNAC(eqx.Module):
    sampling_rate: int = field(static=True)
    encoder_dim: int = field(static=True)
    encoder_rates: list[int] = field(static=True)
    decoder_dim: int | None = field(static=True)
    decoder_rates: list[int] = field(static=True)
    latent_dim: int = field(static=True)
    hop_length: int = field(static=True)
    n_codebooks: int = field(static=True)
    codebook_size: int = field(static=True)
    codebook_dim: int = field(static=True)
    vq_strides: list[int] = field(static=True)
    attn_window_size: int = field(static=True)

    encoder: Encoder
    quantizer: ResidualVectorQuantize
    decoder: Decoder

    def __init__(
        self,
        sampling_rate=44100,
        encoder_dim=64,
        encoder_rates=[3, 3, 7, 7],
        latent_dim=None,
        decoder_dim=1536,
        decoder_rates=[7, 7, 3, 3],
        attn_window_size=32,
        codebook_size=4096,
        codebook_dim=8,
        vq_strides=[8, 4, 2, 1],
        noise=True,
        depthwise=True,
    ):
        self.sampling_rate = sampling_rate
        self.encoder_dim = encoder_dim
        self.encoder_rates = encoder_rates
        self.decoder_dim = decoder_dim
        self.decoder_rates = decoder_rates
        if latent_dim is None:
            latent_dim = encoder_dim * (2 ** len(encoder_rates))
        self.latent_dim = latent_dim
        self.hop_length = np.prod(encoder_rates)
        self.encoder = Encoder(
            encoder_dim,
            encoder_rates,
            depthwise=depthwise,
            attn_window_size=attn_window_size,
        )
        self.n_codebooks = len(vq_strides)
        self.codebook_size = codebook_size
        self.codebook_dim = codebook_dim
        self.vq_strides = vq_strides
        self.attn_window_size = attn_window_size
        self.quantizer = ResidualVectorQuantize(
            input_dim=latent_dim,
            codebook_size=codebook_size,
            codebook_dim=codebook_dim,
            vq_strides=vq_strides,
        )
        self.decoder = Decoder(
            latent_dim,
            decoder_dim,
            decoder_rates,
            noise,
            depthwise=depthwise,
            attn_window_size=attn_window_size,
        )

    def preprocess(self, audio_data):
        length = audio_data.shape[-1]
        lcm = math.lcm(self.vq_strides[0], self.attn_window_size or 1)
        pad_to = self.hop_length * lcm
        right_pad = math.ceil(length / pad_to) * pad_to - length
        audio_data = jnp.pad(audio_data, ((0, 0), (0, right_pad)))  # TODO
        return audio_data

    def __call__(self, audio_data: Array, key=None) -> Tuple[Array, List[Array]]:
        length = audio_data.shape[-1]
        audio_data = self.preprocess(audio_data)
        z = self.encoder(audio_data)
        z_q, codes = self.quantizer(z)
        audio_hat = self.decoder(z_q)
        return audio_hat[..., :length], codes

    def encode(self, audio_data: Array) -> List[Array]:
        audio_data = self.preprocess(audio_data)
        z = self.encoder(audio_data)
        _, codes = self.quantizer(z)
        return codes

    def decode(self, codes: List[Array]) -> Array:
        z_q = self.quantizer.from_codes(codes)
        audio_hat = self.decoder(z_q)
        return audio_hat

    @classmethod
    def from_config(cls, config_path):
        with open(config_path, "r") as f:
            config = json.load(f)
        model = cls(**config)
        return model

    # TODO need to check this whole thing
    @classmethod
    def from_pretrained(cls, repo_id, **kwargs):
        import torch
        from huggingface_hub import hf_hub_download

        config_path = hf_hub_download(repo_id=repo_id, filename="config.json", **kwargs)
        model_path = hf_hub_download(
            repo_id=repo_id, filename="pytorch_model.bin", **kwargs
        )

        ws = torch.load(model_path, map_location="cpu")
        model = cls.from_config(config_path)

        def get_weight(path, model):
            if len(path) == 1:
                weight_name = path[0]
                if isinstance(model, nn.WeightNorm):
                    if weight_name == "bias":
                        return model.layer.bias
                    if weight_name == "original1":  # TODO could be other way around
                        return model.layer.weight
                    if weight_name == "original0":
                        return model.g
                return getattr(model, weight_name)
            if path[0].isdigit():
                return get_weight(path[1:], model[int(path[0])])
            if path[0] == "parametrizations":
                return get_weight(path[2:], model)
            return get_weight(path[1:], getattr(model, path[0]))

        for n, w in ws.items():
            layer = get_weight(n.split("."), model)
            w = jnp.array(w.cpu().numpy())
            if tuple(layer.shape) != tuple(w.shape):
                if w.squeeze().ndim == w.ndim and w.ndim == 3:  # TODO double check this
                    w = jnp.flip(w, axis=2).transpose(1, 0, 2)
                else:
                    w = w.reshape(layer.shape)
            if "alpha" in n:
                w = w.reshape(-1, 1, 1)
            model = eqx.tree_at(lambda m: get_weight(n.split("."), m), model, w)

        return model
