import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib as mpl
import matplotlib.ticker as plticker



def set_size(width = 'neurips', fraction=1, subplots=(1, 1)):
    """Set figure dimensions to avoid scaling in LaTeX.

    Parameters
    ----------
    width: float or string
            Document width in points, or string of predined document type
    fraction: float, optional
            Fraction of the width which you wish the figure to occupy
    subplots: array-like, optional
            The number of rows and columns of subplots.
    Returns
    -------
    fig_dim: tuple
            Dimensions of figure in inches
    """
    if width == 'thesis':
        width_pt = 426.79135
    elif width == 'neurips':
        width_pt = 397.48499
    elif width == 'icml':
        width_pt = 234.8775
    elif width == 'cvpr':
        width_pt = 233.8583
    elif width == 'iccv':
        width_pt = 496.063
    else:
        width_pt = width

    # Width of figure (in pts)
    fig_width_pt = width_pt * fraction
    # Convert from pt to inches
    inches_per_pt = 1 / 72.27

    # Golden ratio to set aesthetic figure height
    # https://disq.us/p/2940ij3
    golden_ratio = (5**.5 - 1) / 2
    # golden_ratio = 0.8

    # Figure width in inches
    fig_width_in = fig_width_pt * inches_per_pt
    # Figure height in inches
    fig_height_in = fig_width_in * golden_ratio * (subplots[0] / subplots[1])

    return (fig_width_in, fig_height_in)


# Using seaborn's style
width = 'iccv'

tex_fonts = {
    # Use LaTeX to write all text
    "text.usetex": False,
    "font.family": "sans-serif",
    # "font.sans-serif": "Helvetica",
    # Use 10pt font in plots, to match 10pt font in document
    "axes.labelsize": 9,
    "font.size": 9,
    # Make the legend/label fonts a little smaller
    "legend.fontsize": 8,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8
}

plt.rcParams.update(tex_fonts)


colors = plt.rcParams['axes.prop_cycle'].by_key()['color']


def _get_fig_axes(nrows, ncols, width, fraction, sharex = 'col', sharey = 'row', layout = 'constrained'):
    fig, axes = plt.subplots(nrows,ncols,figsize = set_size(width=width, fraction = fraction, subplots = (nrows,ncols)), sharex=sharex, sharey = sharey,layout = layout)
    for ax in axes.flat:
        ax.set_xlabel('X Label')
        ax.set_ylabel('Y Label')

        ax.xaxis.set_tick_params(width=0.5)
        ax.yaxis.set_tick_params(width=0.5)

        ax.spines['bottom'].set_linewidth(0.5) # Line width for the bottom axis
        ax.spines['left'].set_linewidth(0.5)
