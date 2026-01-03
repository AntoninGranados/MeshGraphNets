import argparse
from pathlib import Path
import subprocess

import matplotlib.pyplot as plt
import numpy as np


parser = argparse.ArgumentParser(description="Live plot training loss from a log file.")
parser.add_argument("-log", "--log-path", type=Path, help="Path to the loss_log.txt file.", required=True)
parser.add_argument("-remote", "--remote-machine", type=str, help="Remote host to read the log from (default: local)", default=None)
parser.add_argument("-rh", "--remote-home", type=Path, help="Remote base path to resolve the log file against", default=None)
args = parser.parse_args()

plt.style.use("dark_background")
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

while plt.fignum_exists(fignum):
    if args.remote_machine:
        args.log_path.parent.mkdir(parents=True, exist_ok=True)
        if args.remote_home:
            relative_log = Path(args.log_path.as_posix().lstrip("/"))
            remote_log_path = args.remote_home / relative_log
        else:
            remote_log_path = args.log_path
        remote_path = f"{args.remote_machine}:{remote_log_path.as_posix()}"
        subprocess.run(
            ["rsync", "-az", remote_path, str(args.log_path.parent)],
            check=True,
        )
    with open(args.log_path, "r") as file:
        lines = file.readlines()

    loss_terms = []
    for l in lines:
        loss_terms.append(list(map(float, l.strip().split(", "))))
    loss_terms = np.array(loss_terms)
    if loss_terms.size:
        loss_terms[loss_terms == -1] = np.nan
    
    plt.clf()

    max_abs = np.nanmax(np.abs(loss_terms))
    linthresh = max(max_abs * 1e-3, 1e-12)
    gs = fig.add_gridspec(2, 4, height_ratios=[2, 1])

    epochs = np.arange(1, len(loss_terms)+1)

    ax_main = fig.add_subplot(gs[0, :])
    labels = ["$L_{total}$", "$L_{inertia}$", "$L_{gravity}$", "$L_{bending}$", "$L_{stretch}$"]
    label_colors = {}
    for idx, label in enumerate(labels):
        line = ax_main.plot(epochs, loss_terms[:, idx], label=label)[0]
        label_colors[label] = line.get_color()
    ax_main.set_yscale("symlog", linthresh=linthresh)
    ax_main.set_xticks(epochs)
    ax_main.legend()
    ax_main.grid(color="gray", alpha=0.5)

    for i, label in enumerate(labels[1:]):
        ax = fig.add_subplot(gs[1, i])
        ax.plot(epochs, loss_terms[:, i+1], label=label, color=label_colors[label])
        ax.set_yscale("symlog", linthresh=linthresh)
        ax.set_title(label, fontsize=12)
        ax.set_xticks(epochs)
        ax.grid(color="gray", alpha=0.5)

    plt.draw()
    plt.pause(30)
