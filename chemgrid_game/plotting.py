from typing import Iterable
from typing import List

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib.collections import PatchCollection
from matplotlib.patches import Ellipse

sns.set_theme()


def draw_circle(center, radius, scale_x, scale_y, **kwargs):
    return Ellipse((scale_x * center[0], scale_y * center[1]), 2 * scale_x * radius, 2 * scale_y * radius, **kwargs)


def plot_atoms(atoms: np.ndarray, scale: float = 1, fig=None, ax=None, background=False):
    colors = ["white", "red", "green", "blue"]

    h, w = atoms.shape
    if ax is None:
        fig, ax = plt.subplots(figsize=(w * scale, h * scale))
    patches = []
    x_scale, y_scale = 1 / w, 1 / h
    for y in range(h):
        for x in range(w):
            atom = atoms[y, x]
            if atom > 0:
                atom = draw_circle((x + 0.5, y + 0.5), radius=0.4, scale_x=x_scale, scale_y=y_scale, fc=colors[atom])
                patches.append(atom)

                if y + 1 < h and atoms[y + 1, x] > 0:
                    atom = draw_circle((x + 0.5, y + 1), radius=0.1, scale_x=x_scale, scale_y=y_scale, fc='black')
                    patches.append(atom)

                if x + 1 < w and atoms[y, x + 1] > 0:
                    atom = draw_circle((x + 1, y + 0.5), radius=0.1, scale_x=x_scale, scale_y=y_scale, fc='black')
                    patches.append(atom)

    p = PatchCollection(patches, match_original=True)
    ax.add_collection(p)
    ax.invert_yaxis()
    if background:
        ax.set_xticks([])
        ax.set_yticks([])
    else:
        ax.axis("off")

    return fig, ax


def plot_mols(mols: List, m=None, n=1, **kwargs):
    if not isinstance(mols, Iterable):
        mols = [mols]
        m, n = 1, 1
    return plot_atoms_list([m.atoms for m in mols if m is not None], m=m, n=n, **kwargs)


def plot_atoms_list(
        atoms: List[np.ndarray],
        titles: List[str] = (),
        scale=0.8,
        m=None, n=1,
        title: str = None,
        show=True,
        background: bool = False,
        constrained_layout: bool = False
):
    data = np.array(atoms)
    n_imgs, grid_h, grid_w = data.shape
    if m is None:
        m, n = int(np.ceil(np.sqrt(n_imgs))), int(np.ceil(np.sqrt(n_imgs)))

    h, w = m * grid_h * scale, n * grid_w * scale
    fig, axs = plt.subplots(m, n, figsize=(w, h), constrained_layout=constrained_layout)
    if m * n > 1:
        axs = axs.flatten()
    else:
        axs = [axs]

    for i, ax in enumerate(axs):
        if i < len(data):
            plot_atoms(data[i], scale, ax=ax, background=background)
            if i < len(titles):
                ax.set_title(titles[i])
        else:
            ax.axis("off")

    if title:
        fig.suptitle(title)

    if show:
        plt.show()

    return fig, axs
