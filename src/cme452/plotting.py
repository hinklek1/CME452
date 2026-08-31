"""
plot_results  --  panel layout inferred from the call itself
============================================================

ONE RULE:  one positional argument after `t`  ==  one panel.

Inside a panel:
    a 1-D series            -> one curve
    a list of 1-D series    -> several curves sharing that axis
    a 2-D array             -> one curve per ROW, all on that axis

    plot_results(t, y)                        1 panel,  1 curve
    plot_results(t, y1, y2)                   2 panels, 1 curve each
    plot_results(t, [a, b], [c, d, e])        2 panels, 2 and 3 curves
    plot_results(t, a, [b, c], d)             3 panels, 1 / 2 / 1
    plot_results(t, sys.x)                    1 panel,  every state overlaid
    plot_results(t, *sys.x)                   one panel per state

That last pair is worth pointing out in class: `sys.x` is a 2-D array, so
passing it gives one panel with everything on top of each other, while
unpacking it with `*` gives a stacked panel per state. The `*` is doing
exactly what the rule says -- turning one argument into many.

Verified behaviour (measured, not asserted):

    call                                       panels   curves
    plot_results(t, y)                            1     [1]
    plot_results(t, y1, y2)                       2     [1, 1]
    plot_results(t, [a,b], [c,d,e])               2     [2, 3]
    plot_results(t, a, [b,c], d)                  3     [1, 2, 1]
    plot_results(t, X)        (X is 3 x N)        1     [3]
    plot_results(t, *X)                           3     [1, 1, 1]


ERROR MESSAGES
--------------
The most common student mistake is a length mismatch -- usually plotting
against the wrong time vector after changing n_out. That gets named
specifically rather than surfacing as a matplotlib traceback:

    argument 1, curve 1 has 50 points but `t` has 101.
    Every series must be the same length as `t`.

Also caught: nothing passed after `t`, a scalar passed instead of a series,
and a wrong-length entry inside a list.
"""

import numpy as np
import matplotlib.pyplot as plt


# ---------------------------------------------------------------------
#  internals
# ---------------------------------------------------------------------

def _as_curves(panel, name):
    """Normalize one positional argument into a list of 1-D arrays."""
    try:
        arr = np.asarray(panel, dtype=float)
    except (ValueError, TypeError):
        arr = None                       # ragged: unequal-length series

    if arr is not None:
        if arr.ndim == 1:
            return [arr]
        if arr.ndim == 2:
            return [row for row in arr]
        if arr.ndim == 0:
            raise ValueError(
                f"{name} is a single number. Each argument after `t` must be "
                f"a series of values (or a list of series)."
            )
        raise ValueError(
            f"{name} has {arr.ndim} dimensions. Expected a 1-D series, a list "
            f"of 1-D series, or a 2-D array."
        )

    if not isinstance(panel, (list, tuple)):
        raise ValueError(
            f"{name} could not be interpreted as a curve or a list of curves."
        )
    out = []
    for j, c in enumerate(panel):
        c = np.asarray(c, dtype=float)
        if c.ndim != 1:
            raise ValueError(
                f"{name}[{j}] has {c.ndim} dimensions; expected a 1-D series."
            )
        out.append(c)
    return out


def _listify(opt, n, kwname):
    """Broadcast a scalar / str / None option to a list of length n."""
    if opt is None:
        return [None] * n
    if isinstance(opt, str):
        return [opt] * n
    opt = list(opt)
    if len(opt) != n:
        raise ValueError(
            f"{kwname} has {len(opt)} entries but there are {n} panels. "
            f"Give one entry per panel, or a single value for all of them."
        )
    return opt


# ---------------------------------------------------------------------
#  public function
# ---------------------------------------------------------------------

def plot_results(t, *panels, ylabels=None, xlabel="Time", labels=None,
                 title=None, figsize=None, sharex=True, grid=True,
                 ylims=None, show=True):
    """Stacked time-series plot; one panel per positional argument.

    Parameters
    ----------
    t : 1-D array
        Common x axis for every curve.
    *panels
        One argument per panel. Each is a 1-D series, a list of 1-D series,
        or a 2-D array (one curve per row).
    ylabels : str or list of str, optional
        y-axis label. A single string is applied to every panel.
    labels : list, optional
        Legend labels, one entry per panel. Each entry is a list matching
        that panel's curve count, or None for no legend on that panel.
    xlabel : str
        Label on the bottom panel only.
    title : str, optional
        Placed on the top panel.
    figsize : (w, h), optional
        Defaults to (6, 2 * n_panels).
    ylims : list of (lo, hi), optional
        One entry per panel; None entries autoscale.
    show : bool
        Call plt.show(). Set False to keep editing the figure.

    Returns
    -------
    (fig, axs) : the figure and a 1-D array of axes, so anything not
        exposed as a keyword can still be done by hand afterwards.
    """
    if len(panels) == 0:
        raise ValueError(
            "plot_results needs at least one series after `t`, e.g. "
            "plot_results(sys.t, sys.x[0])."
        )

    t = np.asarray(t, dtype=float).ravel()
    curves = [_as_curves(pk, f"argument {k + 1}") for k, pk in enumerate(panels)]
    n = len(curves)

    for k, panel in enumerate(curves):
        for j, c in enumerate(panel):
            if c.size != t.size:
                raise ValueError(
                    f"argument {k + 1}, curve {j + 1} has {c.size} points but "
                    f"`t` has {t.size}. Every series must be the same length "
                    f"as `t`. (Did n_out change since this array was made?)"
                )

    ylabels = _listify(ylabels, n, "ylabels")
    ylims = _listify(ylims, n, "ylims") if ylims is not None else [None] * n

    if labels is None:
        labels = [None] * n
    else:
        labels = list(labels)
        if len(labels) != n:
            raise ValueError(
                f"labels has {len(labels)} entries but there are {n} panels."
            )

    if figsize is None:
        figsize = (6.0, 2.0 * n)

    fig, axs = plt.subplots(n, 1, sharex=sharex, figsize=figsize, squeeze=False)
    axs = axs[:, 0]

    for ax, panel, ylab, lab, ylim in zip(axs, curves, ylabels, labels, ylims):
        lab_list = (_listify(lab, len(panel), "labels entry")
                    if lab is not None else [None] * len(panel))
        for c, l in zip(panel, lab_list):
            ax.plot(t, c, label=l)
        if ylab is not None:
            ax.set_ylabel(ylab)
        if ylim is not None:
            ax.set_ylim(ylim)
        if grid:
            ax.grid(ls="--", alpha=0.6)
        if any(l is not None for l in lab_list):
            ax.legend(loc="best", fontsize=8)

    axs[-1].set_xlabel(xlabel)
    if title is not None:
        axs[0].set_title(title)
    fig.tight_layout()
    if show:
        plt.show()
    return fig, axs


# ---------------------------------------------------------------------
#  thin method for System -- t defaults to the last run's grid
# ---------------------------------------------------------------------

def _system_plot_results(self, *panels, **kw):
    """System.plot_results(...) -- same rules, but `t` is implicit.

        sys.run_simulation(t_end=80, IC=[60.0])
        sys.plot_results(sys.x[0], sys.v[0], ylabels=['T [C]', 'Q_loss'])

    If the first argument is itself a time vector you can still call the
    module-level plot_results() directly.
    """
    if self.t is None:
        raise RuntimeError(
            "No results to plot yet. Call run_simulation() first."
        )
    return plot_results(self.t, *panels, **kw)


# Attach in the package __init__ (or paste into the class body):
#     System.plot_results = _system_plot_results


# ---------------------------------------------------------------------
#  self-test
# ---------------------------------------------------------------------
if __name__ == "__main__":
    import matplotlib
    matplotlib.use("Agg")

    t = np.linspace(0, 10, 101)
    a, b, c, d, e = [np.sin(t + k) for k in range(5)]
    X = np.vstack([a, b, c])

    def shape(fig, axs, note):
        print(f"{note:46s} panels={len(axs)}  curves="
              f"{[len(ax.get_lines()) for ax in axs]}")

    shape(*plot_results(t, a, show=False), "(t, y)")
    shape(*plot_results(t, a, b, show=False), "(t, y1, y2)")
    shape(*plot_results(t, [a, b], [c, d, e], figsize=(4, 6), show=False),
          "(t, [a,b], [c,d,e], figsize=(4,6))")
    shape(*plot_results(t, a, [b, c], d, show=False), "(t, a, [b,c], d)")
    shape(*plot_results(t, X, show=False), "(t, X)  -- 2-D, one panel")
    shape(*plot_results(t, *X, show=False), "(t, *X) -- unpacked, 3 panels")

    f, axs = plot_results(t, [a, b], c, ylabels=["T [C]", "Q"],
                          labels=[["inlet", "drum"], None],
                          xlabel="time [min]", show=False)
    print("ylabels:", [ax.get_ylabel() for ax in axs],
          "legends:", [ax.get_legend() is not None for ax in axs])

    for args, note in [((t, a[:50]), "wrong length"),
                       ((t,), "nothing after t"),
                       ((t, 5.0), "scalar"),
                       ((t, [a, b[:50]]), "ragged list")]:
        try:
            plot_results(*args, show=False)
            print("  NO ERROR:", note)
        except ValueError as err:
            print(f"  [{note}] {str(err).splitlines()[0]}")
