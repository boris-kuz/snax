#!/usr/bin/env python3

from snac_jax.snac import SNAC
import soundfile as sf
import jax.numpy as jnp
import jax
from loguru import logger
from einops import rearrange

from resampy import resample

import equinox as eqx
from functools import partial


@eqx.filter_jit
def transcode(model, input):
    input = jax.lax.stop_gradient(input)
    codes = jax.vmap(model.encode)(input)
    return jax.vmap(model.decode)(codes)


if __name__ == "__main__":
    logger.info(f"{jax.devices()[0]=}")
    model = SNAC.from_pretrained("hubertsiuzdak/snac_44khz")
    y, sr = sf.read("charli.wav")
    logger.info(f"{y.shape=}")
    if sr != 44100:
        y = resample(y, sr, 44100, axis=0)
    y = jnp.asarray(rearrange(y, "t c -> c 1 t"))
    audio_hat = transcode(model, y)
    logger.info(f"{audio_hat.shape=}")
    norm = lambda x: x / jnp.max(jnp.abs(x))
    sf.write("reconstructed.wav", norm(rearrange(y, "c 1 t -> t c")), 44100)
    logger.info("Written to reconstructed.wav")
