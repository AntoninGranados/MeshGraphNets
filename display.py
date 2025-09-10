import numpy as np
import torch
import matplotlib.pyplot as plt

from pathlib import Path

from graph import NodeType, Mesh


def display_trajectory(data: Mesh, meta: dict, max_frame: int|None = None, title: str = "", save_fig: bool = False, save_path: Path|str = "") -> None:
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

        if save_fig:
            plt.savefig(Path(save_path, f"frame_{i:03}.png"), dpi=500)
    
    plt.close()

def display_prediction_target(pred: Mesh, targ: Mesh, meta: dict, max_frame: int|None = None, title: str = "", save_fig: bool = False, save_path: Path|str = "") -> None:
    def get_np(tensor: torch.Tensor) -> np.ndarray:
        """removes batch dimension if there is one and converts to numpy array"""
        if len(tensor.shape) > 3:
            return tensor[0].detach().cpu().numpy()
        return tensor.detach().cpu().numpy()
    
    pred_world_pos: np.ndarray = get_np(pred["world_pos"])
    targ_world_pos: np.ndarray = get_np(targ["world_pos"])

    pred_triangles = get_np(pred["cells"])[0]
    targ_triangles = get_np(targ["cells"])[0]

    # assumes the same number of frames
    frame_count = pred_world_pos.shape[0] if max_frame is None else min(pred_world_pos.shape[0], max_frame)

    plt.style.use('dark_background')
    fig = plt.figure()
    ax1 = fig.add_subplot(121, projection='3d')
    ax2 = fig.add_subplot(122, projection='3d')
    for i in range(frame_count):
        ax1.cla()
        ax1.plot_trisurf(pred_world_pos[i,:,0], pred_world_pos[i,:,1], pred_world_pos[i,:,2], triangles=pred_triangles)
        ax1.set_xlim(np.min(pred_world_pos[:,:,0]), np.max(pred_world_pos[:,:,0]))
        ax1.set_ylim(np.min(pred_world_pos[:,:,1]), np.max(pred_world_pos[:,:,1]))
        ax1.set_zlim(np.min(pred_world_pos[:,:,2]), np.max(pred_world_pos[:,:,2]))
        ax1.set_axis_off()
        ax1.set_title("Prediction")
        ax1.patch.set_edgecolor("w")
        ax1.patch.set_linewidth(1)

        ax2.cla()
        ax2.plot_trisurf(targ_world_pos[i,:,0], targ_world_pos[i,:,1], targ_world_pos[i,:,2], triangles=targ_triangles)
        ax2.set_xlim(np.min(targ_world_pos[:,:,0]), np.max(targ_world_pos[:,:,0]))
        ax2.set_ylim(np.min(targ_world_pos[:,:,1]), np.max(targ_world_pos[:,:,1]))
        ax2.set_zlim(np.min(targ_world_pos[:,:,2]), np.max(targ_world_pos[:,:,2]))
        ax2.set_axis_off()
        ax2.set_title("Target")
        ax2.patch.set_edgecolor("w")
        ax2.patch.set_linewidth(1)

        fig.suptitle(f"{title}\nFrame: {i+1:>4}/{frame_count}")
        fig.tight_layout()
        plt.draw()
        
        plt.pause(meta["dt"])

        if save_fig:
            plt.savefig(Path(save_path, f"frame_{i:03}.png"), dpi=500)
    
    plt.close()
