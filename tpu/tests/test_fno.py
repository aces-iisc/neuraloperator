"""
Correctness net for the fast FNO: the real/imag path must agree with a textbook complex64
jnp.fft implementation.  Runs on CPU, so it guards refactors without needing a TPU.

    .venv/bin/python -m pytest tpu/tests -q
"""
import os
import tempfile

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from ..experimental.fno import reference
from ..fft import core, rfft1d, rfft2d
from ..fft.core import F
from ..fno import layers, model, spectral, train

SHAPES = [(32, 32, 8), (64, 64, 12), (17, 16, 6)]


def _rel(a, b):
    a, b = np.asarray(a), np.asarray(b)
    return float(np.max(np.abs(a - b)) / (np.max(np.abs(b)) + 1e-12))


@pytest.mark.parametrize("n", [64, 256, 1024])
def test_rfft_matches_numpy(n):
    x = jnp.asarray(np.random.default_rng(0).standard_normal((n, 8)), jnp.float32)
    re, im = rfft1d.rfft_full(x)
    want = np.fft.fft(np.asarray(x), axis=0)
    assert _rel(re, want.real) < 1e-4
    assert _rel(im, want.imag) < 1e-4


@pytest.mark.parametrize("n,n_modes", [(256, 4), (256, 16), (1024, 8)])
def test_rfft_low_modes_matches_numpy(n, n_modes):
    x = jnp.asarray(np.random.default_rng(1).standard_normal((n, 8)), jnp.float32)
    re, im = rfft1d.rfft_low_modes(x, n_modes)
    want = np.fft.fft(np.asarray(x), axis=0)[:n_modes]
    assert _rel(re, want.real) < 1e-4
    assert _rel(im, want.imag) < 1e-4


@pytest.mark.parametrize("n_rows,n_cols", [(128, 32), (256, 64)])
def test_rfft2_matches_numpy(n_rows, n_cols):
    field = jnp.asarray(np.random.default_rng(2).standard_normal((n_rows, n_cols, 4)), jnp.float32)
    re, im = rfft2d.rfft2_full(field)
    want = np.fft.fft2(np.asarray(field), axes=(0, 1))[:n_rows // 2]
    assert _rel(re, want.real) < 1e-4
    assert _rel(im, want.imag) < 1e-4


def test_rfft2_staged_cols_matches_direct():
    field = jnp.asarray(np.random.default_rng(3).standard_normal((128, 32, 4)), jnp.float32)
    direct, staged = rfft2d.rfft2_full(field), rfft2d.rfft2_full_staged_cols(field)
    assert _rel(staged[0], direct[0]) < 1e-4
    assert _rel(staged[1], direct[1]) < 1e-4


def test_fft_staged_complex_matches_numpy():
    rng = np.random.default_rng(4)
    re = jnp.asarray(rng.standard_normal((256, 8)), jnp.float32)
    im = jnp.asarray(rng.standard_normal((256, 8)), jnp.float32)
    got_re, got_im = core.fft_staged_complex(re, im, 256)
    want = np.fft.fft(np.asarray(re) + 1j * np.asarray(im), axis=0)
    assert _rel(got_re, want.real) < 1e-4
    assert _rel(got_im, want.imag) < 1e-4


@pytest.mark.parametrize("n_rows,n_cols,n_modes", SHAPES)
def test_spectral_conv_matches_complex64_reference(n_rows, n_cols, n_modes):
    """The whole point of the real/imag rewrite: same answer as the complex64 version."""
    w = layers._init_spectral_weight(jax.random.PRNGKey(0), 4, 4, n_modes)
    x = jnp.asarray(np.random.default_rng(5).standard_normal((2, 4, n_rows, n_cols)), jnp.float32)
    assert _rel(spectral.spectral_conv(w, x, n_modes), reference.spectral_conv(w, x, n_modes)) < 2e-3


def test_forward_matches_reference():
    P = layers.init_fno(jax.random.PRNGKey(7), 1, 1, channels=8, n_modes=6, n_layers=3)
    x = jnp.asarray(np.random.default_rng(6).standard_normal((3, 1, 32, 32)), jnp.float32)
    assert _rel(model.forward(P, x, 6), reference.forward(P, x, 6)) < 2e-3


def test_save_load_round_trip():
    P = layers.init_fno(jax.random.PRNGKey(8), 1, 1, channels=8, n_modes=6, n_layers=2)
    x = jnp.asarray(np.random.default_rng(7).standard_normal((2, 1, 32, 32)), jnp.float32)
    path = os.path.join(tempfile.mkdtemp(), "params.npz")
    model.save_params(P, path)
    restored = model.load_params(path, channels=8, n_modes=6, n_layers=2)
    assert _rel(model.forward(restored, x, 6), model.forward(P, x, 6)) == 0.0


def test_training_reduces_error():
    rng = np.random.default_rng(9)
    x = np.asarray(rng.standard_normal((32, 1, 16, 16)), np.float32)
    y = np.asarray(np.tanh(2 * x), np.float32)
    _, rel, _ = train.train(x, y, x[:8], y[:8], n_modes=4, channels=8, n_layers=2,
                            steps=6, batch=8, log_every=5, verbose=False)
    assert rel < 1.0


@pytest.mark.parametrize("dtype", [jnp.float32, jnp.bfloat16])
def test_dtype_argument_is_threaded_through(dtype):
    """The bf16 path is the fastest engine on device, so the dtype kwarg must reach every
    caller — including the experimental ones that import these under their old aliases."""
    from ..experimental.fft import rfft2_hybrid

    x = jnp.asarray(np.random.default_rng(10).standard_normal((256, 8)), jnp.float32)
    re, im = core.rfft_staged(x, 256, half=True, dtype=dtype)
    assert re.dtype == dtype and im.dtype == dtype

    field = jnp.asarray(np.random.default_rng(11).standard_normal((128, 32, 4)), jnp.float32)
    hyb_re, _ = rfft2_hybrid.rfft2_hybrid(field, 8, 8, dt=dtype)
    exact = np.fft.fft2(np.asarray(field), axes=(0, 1))[:8, :8]
    assert _rel(hyb_re.astype(jnp.float32), exact.real) < (2e-1 if dtype == jnp.bfloat16 else 1e-4)


def test_hybrid_matches_the_main_low_mode_transform():
    """rfft2_hybrid takes a different route to the same modes; it must land in the same place."""
    from ..experimental.fft import rfft2_hybrid

    field = jnp.asarray(np.random.default_rng(12).standard_normal((128, 32, 4)), jnp.float32)
    want_re, want_im = rfft2d.rfft2_low_modes(field, ratio=0.25)
    got_re, got_im = rfft2_hybrid.rfft2_hybrid(field, want_re.shape[0], want_re.shape[1])
    assert _rel(got_re, want_re) < 1e-4
    assert _rel(got_im, want_im) < 1e-4


def test_entry_point_signatures_match_the_bench_call_sites():
    """The benches and experimental modules call these with positional dtype/ratio args.
    Renaming a parameter breaks them silently, since the benches swallow their own errors —
    so pin the exact call shapes they use."""
    from ..fft.core import BF
    from ..fft.rfft1d import rfft_full, rfft_half, rfft_low_modes
    from ..fft.rfft2d import rfft2_full, rfft2_low_modes

    x = jnp.asarray(np.random.default_rng(13).standard_normal((256, 8)), jnp.float32)
    field = jnp.asarray(np.random.default_rng(14).standard_normal((128, 32, 4)), jnp.float32)

    rfft_half(x, F)                              # bench/fft_engines.py, bench_jax.py
    rfft_half(x.astype(BF), BF)                  # bench_jax.py, the bf16 engine
    rfft_full(x)                                 # bench_jax.py
    rfft_low_modes(x, 8)                         # bench/fft_engines.py
    rfft2_full(field, F)                         # bench/fft_engines.py, hybrid2d.py
    rfft2_low_modes(field, 0.5, F)               # bench/fft_engines.py
    rfft2_low_modes(field.astype(BF), 0.5, BF)   # bench_jax.py
