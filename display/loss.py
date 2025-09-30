import matplotlib.pyplot as plt
import numpy as np
import subprocess
import json
from pathlib import Path
from scipy.optimize import curve_fit

if __name__ == "__main__":
    with open(Path(".", "hyperparam.json"), "r") as file:
        hyper = json.loads(file.read())

    device = "gpu2.enst.fr"
    checkpoint_dir = Path(hyper["training"]["checkpoint-dir"])
    train_loss_file = Path(checkpoint_dir, "loss.txt")
    valid_loss_file = Path(checkpoint_dir, "validation_loss.txt")

    plt.style.use('dark_background')
    fig = plt.figure()
    plt.rcParams["font.family"] = "monospace"
    fignum = fig.number

    length = 399*11
    train_loss = []
    valid_loss = []
    while True:
        # Download the loss file from the remote server
        save_file = Path("/", "home", "infres", "agranados-24", "meshgraphnets", checkpoint_dir)
        subprocess.run(["rsync", "--ignore-existing", "-q", f"agranados-24@{device}:{str(Path(save_file, "*"))}", "./checkpoints"])
        subprocess.run(["rsync", "-q", f"agranados-24@{device}:{str(Path(save_file, "loss.txt"))}", "./checkpoints"])
        subprocess.run(["rsync", "-q", f"agranados-24@{device}:{str(Path(save_file, "validation_loss.txt"))}", "./checkpoints"])
        
        if train_loss_file.exists():
            train_loss = np.loadtxt(train_loss_file)

        if valid_loss_file.exists():
            valid_loss = np.loadtxt(valid_loss_file)
        
        # def func(x, a, b, c):
        #     return a * np.exp(-b * x) + c
        # xdata = np.arange(len(train_loss))
        # xdata = xdata[~np.isnan(train_loss)]
        # ydata = train_loss[~np.isnan(train_loss)]
        # param, _ = curve_fit(func, xdata, ydata)

        w = np.ones(2)
        
        n = np.convolve(np.ones_like(valid_loss), w, mode="same")
        valid_loss = np.convolve(valid_loss, w, mode="same") / n

        n = np.convolve(np.ones_like(train_loss), w, mode="same")
        train_loss = np.convolve(train_loss, w, mode="same") / n

        fig.clf()
        ax = fig.add_subplot()

        ax.plot(train_loss, "r", label="Training loss")
        ax.plot(valid_loss, "g", label="Validation loss", lw=1)
        # ax.plot(func(xdata, *param), "c--", label="Loss estimation")
        ax.set_xlabel("Epochs", color="gray")
        ax.set_ylabel("MSE Loss", color="gray")
        ax.semilogy()
        ax.legend()

        ax.patch.set_edgecolor("w")
        ax.patch.set_facecolor("k")
        ax.patch.set_alpha(0.2)
        ax.patch.set_linewidth(1)

        fig.suptitle(f"Model Loss", fontsize=16, y=0.95)
        fig.patch.set_facecolor('k')
        fig.patch.set_alpha(0.2)
        fig.tight_layout()
        plt.draw()
        plt.pause(0.1)

        if not plt.fignum_exists(fignum): break
