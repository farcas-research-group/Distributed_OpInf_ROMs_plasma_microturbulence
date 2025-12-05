import matplotlib as mpl
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import numpy as np
import os

# plot settings
path_env = os.environ.get('PATH', '')
tex_path = '/global/common/software/nersc9/texlive/2024/bin/x86_64-linux'
os.environ['PATH'] = f"{tex_path}:{path_env}"
mpl.use('Agg')  # write-only backend for headless servers
mpl.rcParams.update({
    'figure.dpi': 400,
    'figure.figsize': (7, 3),
    "text.usetex": True, # This will now work
    "font.family": "serif",
    "text.latex.preamble": r"""
        \usepackage{amsmath}
        \usepackage{amssymb}
        """,
    # 'mathtext.fontset': 'dejavuserif', # This setting is ignored when text.usetex=True
    'font.size': 12.5,
    'lines.linewidth': 0, # Note: this seems unusual, are you sure you want no line width?
    'legend.edgecolor': 'none',
    'axes.facecolor': 'w',
    'axes.edgecolor': 'k',
    'axes.labelcolor': 'k',
    'xtick.color': 'k',
    'ytick.color': 'k',
    'savefig.facecolor': 'w'
})


# main plotting function that plots 2d slices of 4D phase space of distribution function
def phase_space_4d(
    true_data,
    norm=False,
    cmap='coolwarm',
    interpolation=False,
    filename='plot_ref.pdf'
):
    """
    2-column version: only plots reference (true) data.
    Layout:
      ┌──────────────────────────────┐
      │  Re[g] (col 0)  │  Im[g] (col 1) │
      ├──────────────────┼────────────────┤
      │        …         │        …       │
      └──────────────────────────────┘
    """
    axis_labels = {
        0: r'$k_x$', 1: r'$z$',
        2: r'$v_{\parallel}$', 3: r'$\mu$'
    }
    axis_bounds = {0: 3, 1: 168, 2: 32, 3: 8}
    axis_pairs = [(0,1), (0,2), (0,3), (1,2), (1,3), (2,3)]
    n_pairs = len(axis_pairs)

    # Create a 6×2 grid
    fig, axes = plt.subplots(
        nrows=n_pairs, ncols=2,
        figsize=(8, 18),
        gridspec_kw={
            'wspace': 0.1,
            'hspace': 0.27,
            'width_ratios': [1, 1],
            'height_ratios': [1]*n_pairs
        },
        constrained_layout=False
    )

    # Column titles (top row only)
    axes[0,0].set_title(r"$\textsc{Gene}$ reference $\mathrm{Re}[\tilde{g}]$", pad=10)
    axes[0,1].set_title(r"$\textsc{Gene}$ reference $\mathrm{Im}[\tilde{g}]$", pad=10)

    for row, (ax_re, ax_im) in enumerate(axes):
        i, j = axis_pairs[row]
        sum_axes = tuple({0,1,2,3} - {i, j})

        # Project data
        t_re = true_data.mean(axis=sum_axes).real
        t_im = true_data.mean(axis=sum_axes).imag

        # Normalize if requested
        if norm:
            mt_re, mt_im = np.abs(t_re).max(), np.abs(t_im).max()
            if mt_re > 0: t_re /= mt_re
            if mt_im > 0: t_im /= mt_im

        vmin = -1 if norm else None
        vmax = 1 if norm else None

        # Plot Re[g]
        im0 = ax_re.imshow(
            t_re,
            aspect='auto',
            origin='lower',
            vmin=vmin, vmax=vmax,
            cmap=cmap,
            interpolation='bicubic' if interpolation else None
        )
        ax_re.set_ylabel(axis_labels[i])
        ax_re.set_xlabel(axis_labels[j])
        ax_re.set_xticks([0, axis_bounds[j]//2 - 1, axis_bounds[j] - 1])
        ax_re.set_xticklabels([1, axis_bounds[j]//2, axis_bounds[j]])
        ax_re.set_yticks([0, axis_bounds[i]//2 - 1, axis_bounds[i] - 1])
        ax_re.set_yticklabels([1, axis_bounds[i]//2, axis_bounds[i]])

        # Plot Im[g]
        im1 = ax_im.imshow(
            t_im,
            aspect='auto',
            origin='lower',
            vmin=vmin, vmax=vmax,
            cmap=cmap,
            interpolation='bicubic' if interpolation else None
        )
        ax_im.set_xlabel(axis_labels[j])
        ax_im.set_xticks([0, axis_bounds[j]//2 - 1, axis_bounds[j] - 1])
        ax_im.set_xticklabels([1, axis_bounds[j]//2, axis_bounds[j]])
        ax_im.set_yticks([])

        # One colorbar for Im[g] column
        divider = make_axes_locatable(ax_im)
        cax = divider.append_axes("right", size="8%", pad=0.08)
        cbar = fig.colorbar(im1, cax=cax)
        if norm:
            cbar.set_ticks([-1, 0, 1])
            cbar.set_ticklabels(['-1', '0', '1'])

    save_filename = f"/global/homes/j/jackk/repos/mcf_turbulence/plots/mpi_test.pdf"
    plt.savefig(save_filename, bbox_inches='tight', dpi=400)
    return fig, axes

r = 1 # reduced dimension
sim_idx = 1 # which training sim to plot
g_mmap = np.load(f"/pscratch/sd/j/jackk/mcf_turbulence/par_output_data_sim{sim_idx}.npy", mmap_mode="r")
t = np.load(f"/pscratch/sd/j/jackk/mcf_turbulence/par_output_times_sim{sim_idx}.npy")

# start_idx, end_idx = 3000, 4730 # what indices to time average over


# Add this line to debug
print(f"Loaded data shape: {g_mmap.shape}") 

# start_idx, end_idx = 300, 473
# start_idx, end_idx = 300, 473


g_mean = np.mean(g_mmap[:, :], axis=1)
g4d    = g_mean.reshape(3, 168, 32, 8, order="F")
del g_mmap, g_mean

save_filename = f"/global/homes/j/jackk/repos/mcf_turbulence/plots/mpi_test.pdf"
phase_space_4d(
    true_data=g4d, 
    norm=True, 
    cmap='coolwarm', 
    interpolation=True, 
    filename=save_filename)