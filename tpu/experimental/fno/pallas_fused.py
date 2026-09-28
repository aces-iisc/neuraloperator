"""One spectral convolution as a single Pallas kernel, so intermediates never reach HBM."""
from functools import partial

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

from ...fft.core import F
from ...fno.spectral import spectral_conv


def _kernel(x_ref, wre_ref, wim_ref, o_ref, *, n_modes, dtype):
    w = {"w_re": wre_ref[...], "w_im": wim_ref[...]}
    o_ref[...] = spectral_conv(w, x_ref[...], n_modes, dtype)


def fused_spectral_conv(weight, x, n_modes, block=1, interpret=False, dtype=F,
                        vmem=None, scratch=None):
    """Real (B, C, H, W) -> (B, C, H, W). Grid over the batch; one sample block does the whole
    transform, mix and inverse in VMEM."""
    B, C, H, W = x.shape
    wre, wim = weight["w_re"], weight["w_im"]
    return pl.pallas_call(
        partial(_kernel, n_modes=n_modes, dtype=dtype),
        grid=(B // block,),
        in_specs=[pl.BlockSpec((block, C, H, W), lambda b: (b, 0, 0, 0)),
                  pl.BlockSpec(wre.shape, lambda b: (0, 0, 0, 0)),
                  pl.BlockSpec(wim.shape, lambda b: (0, 0, 0, 0))],
        out_specs=pl.BlockSpec((block, C, H, W), lambda b: (b, 0, 0, 0)),
        out_shape=jax.ShapeDtypeStruct((B, C, H, W), F),
        compiler_params=pltpu.CompilerParams(
            **({"vmem_limit_bytes": vmem} if vmem else {}),
            **({"internal_scratch_in_bytes": scratch} if scratch else {})),
        interpret=interpret,
    )(x, wre, wim)
