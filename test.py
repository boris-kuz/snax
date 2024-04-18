#!/usr/bin/env python3

from snac_jax.snac import SNAC
import soundfile as sf
import jax.numpy as jnp
import jax

if __name__ == "__main__":
    model = SNAC.from_pretrained("hubertsiuzdak/snac_32khz")
    y, sr = sf.read("chord.wav")
    y = y.reshape(1, 1, -1)
    codes = jax.vmap(model.encode)(jnp.asarray(y))
    audio_hat = jax.vmap(model.decode)(codes)
