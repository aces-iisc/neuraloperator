"""The real transform as one object, configured by the accelerator it runs on.

`FFT` holds the two things the transform needs to know -- which rows to produce and how wide
the matrix unit is -- and exposes the forward and inverse passes the spectral layer uses.  The
radix of a staged sweep is the unit's tile edge, so a 32x32 part stages by 32 without the
transform being edited.

Output frequencies are independent of one another, so evaluating a subset is exact rather than
approximate: `rows` chooses what gets built and nothing outside it is computed.  A staged radix
transform cannot do that -- its butterfly tree shares subexpressions across all N outputs -- so
the two families are different methods, not two settings of one.
"""
import jax.numpy as jnp
import numpy as np

from .core import ACC, F
from ..arch import TPU_V5E


class FFT:
    """Real-input DFT restricted to chosen output rows.

    x is carried as real/imaginary pairs throughout; there is no complex tensor anywhere,
    because the matrix unit has no complex multiply and XLA wraps complex64 in guard code.
    """

    def __init__(self, accel=TPU_V5E, dtype=F):
        self.accel, self.dtype = accel, dtype

    # -- which rows ------------------------------------------------------------------
    @staticmethod
    def band(n_modes, length):
        """Both low-frequency corners: this axis is not conjugate-symmetric."""
        half = n_modes // 2
        return np.concatenate([np.arange(half), np.arange(length - half, length)])

    @staticmethod
    def half_band(n_modes):
        """One side only: the input is real, so the negative half is the conjugate mirror."""
        return np.arange(n_modes // 2)

    # -- the matrices ----------------------------------------------------------------
    def matrix(self, rows, length, sign=-1):
        """(cos, sin) of a length-`length` DFT restricted to `rows`.  sign=-1 forward."""
        k = jnp.asarray(rows)[:, None]
        n = jnp.arange(length)[None, :]
        angle = (sign * 2.0 * jnp.pi / length) * (k * n).astype(F)
        return jnp.cos(angle), jnp.sin(angle)

    def stacked(self, rows, length, sign=-1):
        """cos and sin in one matrix, so one call does the work of two.

        They multiply the same operand, so stacking them along the output axis doubles the rows
        the unit sees per call instead of issuing two narrow matmuls.
        """
        cos, sin = self.matrix(rows, length, sign)
        return jnp.concatenate([cos, sin], axis=0).astype(self.dtype)

    # -- the passes ------------------------------------------------------------------
    def rows_forward(self, x, rows):
        """Transform axis -2 at `rows`.  (n, c, h, w) -> (re, im), each (n, c, |rows|, w)."""
        matrix = self.stacked(rows, x.shape[-2], sign=-1)
        both = jnp.einsum("ah,nchw->ncaw", matrix, x.astype(self.dtype),
                          preferred_element_type=ACC)
        keep = len(rows)
        return both[:, :, :keep], both[:, :, keep:]

    def cols_forward(self, re, im, cols, length):
        """Transform axis -1 at `cols`, on a complex input, giving the kept mode block."""
        matrix = self.stacked(cols, length, sign=-1)
        over = lambda z: jnp.einsum("dw,ncaw->ncad", matrix, z, preferred_element_type=ACC)
        from_re, from_im = over(re), over(im)
        keep = len(cols)
        return (from_re[..., :keep] - from_im[..., keep:],
                from_re[..., keep:] + from_im[..., :keep])

    def cols_inverse(self, re, im, cols, length):
        """Columns back, folding in the Hermitian partners the one-sided axis left out."""
        fold = jnp.where(jnp.asarray(cols) == 0, 1.0, 2.0).astype(F)   # DC once, the rest twice
        cos, sin = self.matrix(cols, length, sign=+1)
        cos, sin = (cos * fold[:, None]).T, (sin * fold[:, None]).T
        block = jnp.concatenate([jnp.concatenate([cos, -sin], axis=1),
                                 jnp.concatenate([sin, cos], axis=1)], axis=0)
        # The inverse matrices stay in f32: by this point the data is f32 too -- the channel
        # mix accumulates there -- so narrowing them would cost accuracy and save no traffic.
        out = jnp.einsum("wd,noad->noaw", block,
                         jnp.concatenate([re, im], axis=-1), preferred_element_type=ACC)
        return out[..., :length], out[..., length:]

    def rows_inverse(self, re, im, rows, length):
        """Rows back.  The field is real, so only the real part survives and the two halves
        fold into a single matmul rather than two."""
        cos, sin = self.matrix(rows, length, sign=+1)
        block = jnp.concatenate([cos, -sin], axis=0)        # f32, as above
        return jnp.einsum("ha,noaw->nohw", block.T,
                          jnp.concatenate([re, im], axis=2), preferred_element_type=ACC)

    # -- staging, for a transform that keeps the whole spectrum -----------------------
    def stages(self, length):
        """How a length-`length` sweep decomposes on this unit: one tile-wide stage at a time."""
        radix, out = self.accel.radix, []
        while length > radix and length % radix == 0:
            out.append(radix)
            length //= radix
        return out, length

    def __repr__(self):
        return f"<FFT radix={self.accel.radix} dtype={self.dtype.__name__}>"
