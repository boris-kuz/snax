import math

import equinox as eqx
import equinox.nn as nn
import jax
import jax.numpy as jnp

from jaxtyping import Array
from typing import List

from .attention import LocalMHA
from .utils import pseudo_rn

from loguru import logger


def WNConv1d(*args, **kwargs):
    return nn.WeightNorm(nn.Conv1d(*args, **kwargs, key=pseudo_rn()))


def WNConvTranspose1d(*args, **kwargs):
    return nn.WeightNorm(nn.ConvTranspose1d(*args, **kwargs, key=pseudo_rn()), axis=1)


class Encoder(eqx.Module):
    block: nn.Sequential

    def __init__(
        self,
        d_model=64,
        strides=[3, 3, 7, 7],
        depthwise=False,
        attn_window_size=32,
    ):
        layers = [WNConv1d(1, d_model, kernel_size=7, padding=3)]
        for stride in strides:
            d_model *= 2
            groups = d_model // 2 if depthwise else 1
            layers += [EncoderBlock(output_dim=d_model, stride=stride, groups=groups)]
        if attn_window_size is not None:
            layers += [LocalMHA(dim=d_model, window_size=attn_window_size)]
        groups = d_model if depthwise else 1
        layers += [
            WNConv1d(d_model, d_model, kernel_size=7, padding=3, groups=groups),
        ]
        self.block = nn.Sequential(layers)

    def __call__(self, x, key=None):
        return self.block(x)


class Decoder(eqx.Module):
    model: nn.Sequential

    def __init__(
        self,
        input_channel,
        channels,
        rates,
        noise=False,
        depthwise=False,
        attn_window_size=32,
        d_out=1,
    ):

        if depthwise:
            layers = [
                WNConv1d(
                    input_channel,
                    input_channel,
                    kernel_size=7,
                    padding=3,
                    groups=input_channel,
                ),
                WNConv1d(input_channel, channels, kernel_size=1),
            ]
        else:
            layers = [WNConv1d(input_channel, channels, kernel_size=7, padding=3)]

        if attn_window_size is not None:
            layers += [LocalMHA(dim=channels, window_size=attn_window_size)]

        for i, stride in enumerate(rates):
            input_dim = channels // 2**i
            output_dim = channels // 2 ** (i + 1)
            groups = output_dim if depthwise else 1
            layers += [
                DecoderBlock(input_dim, output_dim, stride, noise, groups=groups)
            ]

        layers += [
            Snake1d(output_dim),
            WNConv1d(output_dim, d_out, kernel_size=7, padding=3),
            nn.Lambda(jnp.tanh),
        ]
        self.model = nn.Sequential(layers)

    def __call__(self, x, key=None):
        x = self.model(x)
        return x


class ResidualUnit(eqx.Module):
    block: nn.Sequential

    def __init__(self, dim=16, dilation=1, kernel=7, groups=1):
        pad = ((kernel - 1) * dilation) // 2
        self.block = nn.Sequential(
            [
                Snake1d(dim),
                WNConv1d(
                    dim,
                    dim,
                    kernel_size=kernel,
                    dilation=dilation,
                    padding=pad,
                    groups=groups,
                ),
                Snake1d(dim),
                WNConv1d(dim, dim, kernel_size=1),
            ]
        )

    def __call__(self, x, key=None):
        y = self.block(x)
        pad = (x.shape[-1] - y.shape[-1]) // 2
        if pad > 0:
            x = x[..., pad:-pad]
        return x + y


class EncoderBlock(eqx.Module):
    block: nn.Sequential

    def __init__(self, output_dim=16, input_dim=None, stride=1, groups=1):
        input_dim = input_dim or output_dim // 2
        self.block = nn.Sequential(
            [
                ResidualUnit(input_dim, dilation=1, groups=groups),
                ResidualUnit(input_dim, dilation=3, groups=groups),
                ResidualUnit(input_dim, dilation=9, groups=groups),
                Snake1d(input_dim),
                WNConv1d(
                    input_dim,
                    output_dim,
                    kernel_size=2 * stride,
                    stride=stride,
                    padding=math.ceil(stride / 2),
                ),
            ]
        )

    def __call__(self, x, key=None):
        return self.block(x)


class NoiseBlock(eqx.Module):
    linear: nn.WeightNorm

    def __init__(self, dim):
        self.linear = WNConv1d(dim, dim, kernel_size=1, use_bias=False)

    def __call__(self, x, key=None):
        C, T = x.shape
        noise = jax.random.normal(shape=(1, T), key=pseudo_rn())
        h = self.linear(x)
        n = noise * h
        x = x + n
        return x


class DecoderBlock(eqx.Module):
    block: nn.Sequential

    def __init__(self, input_dim=16, output_dim=8, stride=1, noise=False, groups=1):
        layers = [
            Snake1d(input_dim),
            WNConvTranspose1d(
                input_dim,
                output_dim,
                kernel_size=2 * stride,
                stride=stride,
                padding=math.ceil(stride / 2),
                output_padding=stride % 2,
            ),
        ]
        if noise:
            layers.append(NoiseBlock(output_dim))
        layers.extend(
            [
                ResidualUnit(output_dim, dilation=1, groups=groups),
                ResidualUnit(output_dim, dilation=3, groups=groups),
                ResidualUnit(output_dim, dilation=9, groups=groups),
            ]
        )
        self.block = nn.Sequential(layers)

    def __call__(self, x, key=None):
        return self.block(x)


@jax.jit
def snake(x, alpha):
    shape = x.shape
    x = x.reshape(shape[0], shape[1], -1)
    x = x + (jnp.sin(alpha * x) ** 2) / (alpha + 1e-9)
    x = x.reshape(shape)
    return x


class Snake1d(eqx.Module):
    alpha: Array

    def __init__(self, channels):
        self.alpha = jnp.ones((1, channels, 1))

    @eqx.filter_jit
    def __call__(self, x, key=None):
        return snake(x, self.alpha)
