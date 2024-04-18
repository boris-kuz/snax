#!/usr/bin/env python3

from snax.snax import SNAX
from snax.utils import timer
import soundfile as sf
import jax.numpy as jnp
import jax
from loguru import logger
from einops import rearrange

from resampy import resample

import equinox as eqx
from functools import partial

import numpy as np


@eqx.filter_jit
def transcode(model, input):
    input = jax.lax.stop_gradient(input)
    codes = jax.vmap(model.encode)(input)
    return jax.vmap(model.decode)(codes)


def get_data():
    y, sr = sf.read("charli.wav")
    if sr != 44100:
        y = resample(y, sr, 44100, axis=0)
    y = jnp.asarray(rearrange(y, "t c -> c 1 t"))
    return y


def get_random_data(l):
    y = np.random.randn(l, 1)
    y = jnp.asarray(rearrange(y, "t c -> c 1 t"))
    return y


if __name__ == "__main__":
    model = eqx.nn.inference_mode(SNAX.from_pretrained("hubertsiuzdak/snac_44khz"))
    audio = get_random_data(441000)
    y = np.random.randn(*audio.shape)
    audio_hat = transcode(model, y)  # trigger jit&trace
    y = audio
    with timer:
        audio_hat = transcode(model, y)
    logger.info(f"Took {timer.elapsed_time:.2e}s")
    norm = lambda x: x / jnp.max(jnp.abs(x))
    sf.write("reconstructed.wav", norm(rearrange(y, "c 1 t -> t c")), 44100)
