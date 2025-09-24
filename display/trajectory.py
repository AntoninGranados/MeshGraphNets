import numpy as np
import torch
import matplotlib.pyplot as plt

from pathlib import Path

from graph import NodeType, Mesh

def display_trajectory(data: Mesh, meta: dict, max_frame: int|None = None, title: str = "", save: bool = False, save_path: Path|str = "") -> None:
    def get_np(tensor: torch.Tensor) -> np.ndarray:
        """removes batch dimension if there is one and converts to numpy array"""
        if len(tensor.shape) > 3:
            return tensor[0].detach().cpu().numpy()
        return tensor.detach().cpu().numpy()

    world_pos: np.ndarray = get_np(data["world_pos"])

    movement: np.ndarray = get_np(data["world_pos"]) - get_np(data["prev|world_pos"])
    movement = np.square(movement) * 500 + 0.1
    movement = np.minimum(movement, 1)
    movement = np.maximum(movement, 0)

    node_type: np.ndarray = get_np(data["node_type"])[0].flatten()
    size = np.array([1, 0, 0, 20])[node_type]
    color = np.where(
        np.repeat(np.repeat(np.expand_dims(node_type == NodeType.HANDLE, axis=[0,2]), movement.shape[0], axis=0), 3, axis=2),
        np.repeat(np.repeat(np.expand_dims([1, 0, 0], axis=[0,1]), movement.shape[0], axis=0), movement.shape[1], axis=1),
        movement
    )
    triangles = get_np(data["cells"])[0]

    plt.style.use('dark_background')
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')

    frame_count = movement.shape[0] if max_frame is None else min(movement.shape[0], max_frame)
    for i in range(frame_count):
        plt.cla()
        ax.plot_trisurf(world_pos[i,:,0], world_pos[i,:,1], world_pos[i,:,2], triangles=triangles, color=(0.,0.,0.,0.), edgecolor=(1.,1.,1.,0.3), linewidth=0.5)
        ax.scatter(world_pos[i,:,0], world_pos[i,:,1], world_pos[i,:,2], s=size, c=color[i], alpha=0.5) # type: ignore
        ax.set_xlim(np.min(world_pos[:,:,0]), np.max(world_pos[:,:,0]))
        ax.set_ylim(np.min(world_pos[:,:,1]), np.max(world_pos[:,:,1]))
        ax.set_zlim(np.min(world_pos[:,:,2]), np.max(world_pos[:,:,2]))
        ax.set_axis_off()
        ax.set_title(f"{title}\nFrame: {i+1:>4}/{frame_count}")
        plt.tight_layout()
        plt.draw()
        
        plt.pause(meta["dt"])

        if save:
            plt.savefig(Path(save_path, f"frame_{i:03}.png"), dpi=500)
    
    plt.close()

def display_trajectory_list(meshes: list[Mesh], names: list[str], meta: dict, title: str = "", loss: list = [], save: bool = False, save_path: Path|str = "") -> None:
    def get_np(tensor: torch.Tensor) -> np.ndarray:
        """removes batch dimension if there is one and converts to numpy array"""
        if len(tensor.shape) > 3:
            return tensor[0].detach().cpu().numpy()
        return tensor.detach().cpu().numpy()

    world_pos = np.array([get_np(mesh["world_pos"]) for mesh in meshes])
    triangles = np.array([get_np(mesh["cells"])[0] for mesh in meshes])

    plt.style.use('dark_background')
    fig, axs = plt.subplots(1, len(meshes), subplot_kw=dict(projection='3d'))
    fig.set_size_inches((3*len(meshes), 4))
    fig.subplots_adjust(left=0.05, bottom=0, right=0.95, top=0.80)
    plt.rcParams["font.family"] = "monospace"

    frame_count = world_pos[0].shape[0]
    for i in range(frame_count):
        fig.clf()
        axs = [fig.add_subplot(1, len(meshes), a+1, projection='3d') for a in range(len(meshes))]
        for a in range(len(meshes)):
            # axs[a].cla()
            axs[a].plot_trisurf(world_pos[a,i,:,0], world_pos[a,i,:,1], world_pos[a,i,:,2], triangles=triangles[a], color="white", alpha=1)
            axs[a].set_xlim(np.min(world_pos[:,:,:,0]), np.max(world_pos[:,:,:,0]))
            axs[a].set_ylim(np.min(world_pos[:,:,:,1]), np.max(world_pos[:,:,:,1]))
            axs[a].set_zlim(np.min(world_pos[:,:,:,2]), np.max(world_pos[:,:,:,2]))
            axs[a].set_axis_off()
            axs[a].set_aspect('equal')

            axs[a].set_title(f"{names[a]}", color="gray")
            axs[a].patch.set_edgecolor("w")
            axs[a].patch.set_facecolor("k")
            axs[a].patch.set_alpha(0.2)
            axs[a].patch.set_linewidth(1)

        subtitle = f"Frame: {i+1:>3}/{frame_count}"
        if loss != []:
            subtitle = f"Frame: {i+1:>3}/{frame_count}\nLoss (RMSE): {loss[i]:.4f}"
        fig.suptitle(f"{title}", fontsize=16, y=0.95)
        fig.text(0.5, 0.80, subtitle, ha="center", fontsize=10, color="white")
        fig.patch.set_facecolor('k')
        fig.patch.set_alpha(0.2)

        # fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.9))
        plt.draw()
        plt.pause(meta["dt"])

        if save:
            plt.savefig(Path(save_path, f"frame_{i:03}.png"), dpi=500, transparent=True)
    
    plt.close()
    
def display_prediction_target(pred: Mesh, targ: Mesh, meta: dict, title: str = "", loss: list = [], save: bool = False, save_path: Path|str = "") -> None:
    display_trajectory_list([pred, targ], ["Prediction", "Target"], meta, title, loss, save, save_path)
