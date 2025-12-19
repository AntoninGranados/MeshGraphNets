import matplotlib.pyplot as plt
import numpy as np
import subprocess
import json
from pathlib import Path
from scipy.optimize import curve_fit


plt.style.use('dark_background')
fig = plt.figure(constrained_layout=True)
plt.rcParams["font.family"] = "monospace"
fignum = fig.number

while plt.fignum_exists(fignum):
    file = open(Path(".", "checkpoints", "flag-gravity-test", "loss_log.txt"), "r")
    lines = file.readlines()
    file.close()

    loss_terms = []
    for l in lines[1:]:
        loss_terms.append(list(map(float, l.strip().split(", "))))
    loss_terms = np.array(loss_terms)


    plt.clf()
    plt.plot(loss_terms[:, 0], "r", label="Total Loss")
    plt.semilogy()

    """
    gs = fig.add_gridspec(2, 4, height_ratios=[2, 1])

    ax_main = fig.add_subplot(gs[0, :])
    labels = ["Total Loss", "L_inertia", "L_gravity", "L_bending", "L_stretch"]
    for idx, label in enumerate(labels):
        ax_main.plot(loss_terms[:, idx], label=label)
    # ax_main.semilogy()
    ax_main.legend()

    for i, (idx, label) in enumerate(zip(range(1, 5), labels[1:])):
        ax = fig.add_subplot(gs[1, i])
        ax.plot(loss_terms[:, idx], label=label)
        # ax.semilogy()
        ax.set_title(label, fontsize=8)
        ax.tick_params(axis="both", which="both", labelsize=7)
    """

    plt.draw()
    plt.pause(30)

