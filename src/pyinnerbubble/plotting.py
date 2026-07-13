"""MATLAB-like figure styling for the library's plots.

Call :func:`use_matlab_style` once (before plotting) to give every subsequent matplotlib
figure a MATLAB (LaTeX-interpreter) look: Computer Modern for all text -- ticks, labels and
math -- with outward tick marks and thin black axes, as in the paper figures. Colormaps are
left untouched. Uses only fonts bundled with matplotlib, so nothing needs to be installed.
"""

from __future__ import annotations

import functools

import matplotlib as mpl

# MATLAB (LaTeX-interpreter) look, using ONLY fonts that ship with matplotlib -- no install
# needed. Computer Modern (cmr10) is used for ALL text (ticks, labels, titles) and math, so
# tick numbers match the LaTeX-style symbols. Outward ticks, thin black frame, white bg.
_MATLAB_RC = {
    "font.family": "cmr10",              # Computer Modern -- bundled with matplotlib, no install
    "mathtext.fontset": "cm",            # matching LaTeX-style math
    "axes.formatter.use_mathtext": True,  # render tick numbers in the same (CM) font
    "axes.unicode_minus": False,          # cmr10 has no Unicode minus glyph -> use ASCII hyphen
    # ONE size drives all figure text -- ticks, axis labels, titles AND the colorbar
    # (the plot functions size the colorbar from this same rcParams["font.size"]).
    "font.size": 18,
    "axes.linewidth": 0.8,
    "axes.titlesize": "medium",
    "axes.labelsize": "medium",
    "xtick.direction": "out",
    "ytick.direction": "out",
    "xtick.major.size": 5.0,
    "ytick.major.size": 5.0,
    "xtick.major.width": 0.8,
    "ytick.major.width": 0.8,
    "xtick.top": False,
    "ytick.right": False,
    "figure.facecolor": "white",
    "savefig.facecolor": "white",
    "lines.linewidth": 1.2,
    "figure.dpi": 120,
}


def use_matlab_style() -> None:
    """Apply the MATLAB-like look to all subsequent matplotlib figures.

    Call once at the top of a script (before any plotting)::

        from pyinnerbubble import use_matlab_style
        use_matlab_style()

    Notes
    -----
    * Uses Helvetica/Arial if installed, else falls back to a similar sans-serif.
    * Wrap symbols in ``$...$`` in your titles/labels to get the LaTeX-style math
      (e.g. ``r"$|u|$"`` or ``r"$(\\sigma^2_{|u|}/\\sigma^2_{max})$"``).
    """
    mpl.rcParams.update(_MATLAB_RC)


def matlab_styled(func):
    """Decorator that renders a plotting function's figure under the MATLAB style.

    The style is applied via ``matplotlib.rc_context`` (scoped to the call), so it does NOT
    permanently change global rcParams -- every library plot comes out styled without the
    caller having to invoke :func:`use_matlab_style`.
    """
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        import matplotlib.pyplot as plt
        with plt.rc_context(_MATLAB_RC):
            return func(*args, **kwargs)
    return wrapper
