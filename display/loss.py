import matplotlib.pyplot as plt
import numpy as np
import subprocess
import json
from pathlib import Path
from scipy.optimize import curve_fit
from matplotlib.ticker import FuncFormatter


plt.style.use('dark_background')
fig = plt.figure(constrained_layout=True)
plt.rcParams["font.family"] = "monospace"
fignum = fig.number

def _clamp_tick(value, _pos, max_chars=7):
    if not np.isfinite(value):
        return ""
    if value == 0:
        return "0"
    text = f"{value:.3g}"
    if len(text) > max_chars:
        text = f"{value:.0e}"
    if len(text) > max_chars:
        text = text[:max_chars]
    return text

tick_formatter = FuncFormatter(_clamp_tick)

while plt.fignum_exists(fignum):
    file = open(Path(".", "checkpoints", "flag-gravity-test", "loss_log.txt"), "r")
    lines = file.readlines()
    file.close()

    loss_terms = []
    for l in lines[1:]:
        loss_terms.append(list(map(float, l.strip().split(", "))))
    loss_terms = np.array(loss_terms)
    if loss_terms.size:
        loss_terms[loss_terms == -1] = np.nan

    """
    plt.clf()
    plt.plot(loss_terms[:, 0], "r", label="Total Loss")
    plt.semilogy()
    """
    
    plt.clf()

    max_abs = np.nanmax(np.abs(loss_terms))
    linthresh = max(max_abs * 1e-3, 1e-12)
    gs = fig.add_gridspec(2, 4, height_ratios=[2, 1])

    ax_main = fig.add_subplot(gs[0, :])
    labels = ["Total Loss", "L_inertia", "L_gravity", "L_bending", "L_stretch"]
    label_colors = {}
    for idx, label in enumerate(labels):
        line = ax_main.plot(loss_terms[:, idx], label=label)[0]
        label_colors[label] = line.get_color()
    ax_main.set_yscale("symlog", linthresh=linthresh)
    ax_main.yaxis.set_major_formatter(tick_formatter)
    ax_main.legend()

    for i, label in enumerate(labels[1:]):
        ax = fig.add_subplot(gs[1, i])
        ax.plot(loss_terms[:, i+1], label=label, color=label_colors[label])
        ax.set_yscale("symlog", linthresh=linthresh)
        ax.yaxis.set_major_formatter(tick_formatter)
        ax.set_title(label, fontsize=8)
        ax.tick_params(axis="both", which="both", labelsize=7)

    plt.draw()
    plt.pause(30)
