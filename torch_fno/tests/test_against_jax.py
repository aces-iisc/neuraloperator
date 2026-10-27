"""The port is only correct if it reproduces the reference, so that is what is asserted.

Every test copies weights from the JAX tree into the torch module and compares outputs. A
tolerance of 1e-5 is f32 round-off at these depths; anything structural shows up far above it.
"""
import numpy as np
import pytest
import torch

jax = pytest.importorskip("jax")
import jax.numpy as jnp  # noqa: E402

from tpu.fft.core import F as JF  # noqa: E402
from tpu.fno import layers as jl, model as jm  # noqa: E402
from tpu.fno.spectral import cp_conv, separable_conv, spectral_conv  # noqa: E402
from torch_fno.fno.model import FNO  # noqa: E402
from torch_fno.fno.spectral import (CPSpectralConv, SeparableSpectralConv,  # noqa: E402
                                    SpectralConv)

TOL = 1e-5


def rel(a, b):
    return float(np.abs(a - b).max() / max(np.abs(b).max(), 1e-30))


def put(param, value):
    with torch.no_grad():
        param.copy_(torch.from_numpy(np.asarray(value).copy()))


def field(batch, ch, n, seed=0):
    return np.random.default_rng(seed).standard_normal((batch, ch, n, n)).astype(np.float32)


@pytest.mark.parametrize("channels,n_modes,grid", [(8, 16, 32), (16, 24, 64)])
def test_dense_matches_jax(channels, n_modes, grid):
    jw = jl._init_spectral_weight(jax.random.PRNGKey(0), channels, channels, n_modes)
    mod = SpectralConv(channels, channels, n_modes)
    put(mod.w_re, jw["w_re"])
    put(mod.w_im, jw["w_im"])
    x = field(2, channels, grid)
    want = np.asarray(spectral_conv(jw, jnp.asarray(x), n_modes, JF))
    assert rel(mod(torch.from_numpy(x)).detach().numpy(), want) < TOL


def test_prebuilt_block_changes_nothing():
    channels, n_modes = 8, 16
    mod = SpectralConv(channels, channels, n_modes)
    x = torch.from_numpy(field(2, channels, 32))
    before = mod(x).detach().numpy()
    after = mod.prebuild()(x).detach().numpy()
    assert rel(after, before) < 1e-6


def test_separable_matches_jax():
    channels, n_modes = 8, 16
    jw = jl.init_separable_weight(jax.random.PRNGKey(1), channels, channels, n_modes)
    mod = SeparableSpectralConv(channels, channels, n_modes)
    put(mod.k_block, jw["k_block"])
    put(mod.l_block, jw["l_block"])
    x = field(2, channels, 32)
    want = np.asarray(separable_conv(jw, jnp.asarray(x), n_modes, JF))
    assert rel(mod(torch.from_numpy(x)).detach().numpy(), want) < TOL


@pytest.mark.parametrize("rank", [2, 8])
def test_cp_matches_jax(rank):
    channels, n_modes = 8, 16
    jw = jl.init_cp_weight(jax.random.PRNGKey(2), channels, channels, n_modes, rank)
    mod = CPSpectralConv(channels, channels, n_modes, rank=rank)
    for key, value in jw.items():
        put(getattr(mod, key), value)
    x = field(2, channels, 32)
    want = np.asarray(cp_conv(jw, jnp.asarray(x), n_modes, JF))
    assert rel(mod(torch.from_numpy(x)).detach().numpy(), want) < TOL


def _port_model(mod, tree):
    for name, src in (("lift", tree["lift"]), ("project", tree["proj"])):
        target = getattr(mod, name)
        for part in ("l1", "l2"):
            put(getattr(target, part).weight, src[part]["W"])
            put(getattr(target, part).bias, src[part]["b"])
    for block, jb in zip(mod.blocks, tree["blocks"]):
        put(block.spectral.w_re, jb["filter"]["w_re"])
        put(block.spectral.w_im, jb["filter"]["w_im"])
        put(block.skip.weight, jb["skip"]["W"])
        put(block.skip.bias, jb["skip"]["b"])


@pytest.mark.parametrize("channels,n_modes,n_layers,grid",
                         [(8, 16, 2, 32), (24, 16, 4, 32), (32, 24, 4, 64)])
def test_whole_model_matches_jax(channels, n_modes, n_layers, grid):
    tree = jl.init_fno(jax.random.PRNGKey(0), 1, 1, channels, n_modes, n_layers)
    mod = FNO(1, 1, channels, n_modes, n_layers)
    _port_model(mod, tree)
    assert mod.n_params() == jl.n_params(tree)
    x = field(2, 1, grid, seed=1)
    want = np.asarray(jm.forward(tree, jnp.asarray(x), n_modes, JF))
    assert rel(mod(torch.from_numpy(x)).detach().numpy(), want) < TOL


def test_losses_match_jax():
    pred = field(4, 1, 32, seed=3)
    y = field(4, 1, 32, seed=4)
    tp, ty = torch.from_numpy(pred), torch.from_numpy(y)
    from torch_fno.fno.layers import h1_rel, lp_rel
    assert abs(float(lp_rel(tp, ty)) - float(jl.lp_rel(jnp.asarray(pred), jnp.asarray(y)))) < 1e-4
    assert abs(float(h1_rel(tp, ty)) - float(jl.h1_rel(jnp.asarray(pred), jnp.asarray(y)))) < 1e-3


def test_gradients_flow():
    mod = FNO(1, 1, 8, 16, 2)
    x = torch.from_numpy(field(2, 1, 32, seed=5))
    mod(x).pow(2).mean().backward()
    missing = [n for n, p in mod.named_parameters() if p.grad is None or not p.grad.any()]
    assert not missing, f"no gradient reached: {missing}"


@pytest.mark.parametrize("cls,kw", [(SpectralConv, {}), (SeparableSpectralConv, {}),
                                    (CPSpectralConv, {"rank": 4})])
@pytest.mark.parametrize("grid,n_modes", [(32, 16), (64, 24)])
def test_layouts_agree(cls, kw, grid, n_modes):
    """mode-major rearranges the contractions; it must not change the answer."""
    channels = 8
    mod = cls(channels, channels, n_modes, **kw)
    x = torch.from_numpy(field(3, channels, grid, seed=7))
    mod.layout = "sample"
    want = mod(x).detach().numpy()
    mod.layout = "mode"
    got = mod(x).detach().numpy()
    assert rel(got, want) < 1e-5, f"layouts disagree by {rel(got, want):.2e}"


def test_mode_major_matches_jax():
    channels, n_modes, grid = 8, 16, 32
    jw = jl._init_spectral_weight(jax.random.PRNGKey(0), channels, channels, n_modes)
    mod = SpectralConv(channels, channels, n_modes)
    put(mod.w_re, jw["w_re"])
    put(mod.w_im, jw["w_im"])
    mod.layout = "mode"
    x = field(2, channels, grid)
    want = np.asarray(spectral_conv(jw, jnp.asarray(x), n_modes, JF))
    assert rel(mod(torch.from_numpy(x)).detach().numpy(), want) < TOL


def test_whole_model_mode_major_matches_jax():
    channels, n_modes, n_layers, grid = 24, 16, 4, 32
    tree = jl.init_fno(jax.random.PRNGKey(0), 1, 1, channels, n_modes, n_layers)
    mod = FNO(1, 1, channels, n_modes, n_layers)
    _port_model(mod, tree)
    for block in mod.blocks:
        block.spectral.layout = "mode"
    x = field(2, 1, grid, seed=1)
    want = np.asarray(jm.forward(tree, jnp.asarray(x), n_modes, JF))
    assert rel(mod(torch.from_numpy(x)).detach().numpy(), want) < TOL
