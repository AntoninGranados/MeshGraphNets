import matplotlib.pyplot as plt
import numpy as np
import subprocess
import json
from pathlib import Path
from scipy.optimize import curve_fit


with open(Path(".", "hyperparam.json"), "r") as file:
    hyper = json.loads(file.read())

device = "gpu2.enst.fr"
checkpoint_dir = Path(hyper["training"]["checkpoint-dir"])
train_loss_file = Path(checkpoint_dir, "loss.txt")
valid_loss_file = Path(checkpoint_dir, "validation_loss.txt")

plt.style.use('dark_background')

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
        train_loss = list(np.loadtxt(train_loss_file))

    if valid_loss_file.exists():
        valid_loss = list(np.loadtxt(valid_loss_file))

    # def func(x, a, b, c, d):
    #     return a * np.exp(-b * x) + c
    # xdata = np.arange(len(train_loss))
    # param, _ = curve_fit(func, xdata, train_loss)

    plt.clf()
    plt.tight_layout()
    plt.plot(valid_loss, "g", label="Validation loss", lw=1)
    plt.plot(train_loss, "r", label="Training loss")
    # plt.plot(func(xdata, *param), "c--", label="Loss estimation")
    plt.semilogy()
    plt.legend()
    plt.draw()
    plt.pause(0.1)
